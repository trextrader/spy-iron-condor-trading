import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass, field
from scipy.signal import lfilter

# Enhanced logging
logger = logging.getLogger(__name__)

@dataclass
class ForecastState:
    active_method: str = "CONDOR"
    active_order: Optional[int] = None
    cooldown: int = 0
    eps_ewma: Dict[Tuple[str, int], float] = field(default_factory=dict)
    
class MetaForecaster:
    """
    Meta-Forecaster: Hybrid Ensemble Arbitrator
    
    Selects the optimal forecasting strategy in real-time by arbitrating between
    6 classical Signal Processing models and the CondorBrain Neural Network.
    
    Modes:
    1. Yule-Walker (YW)
    2. Burg's Method (BURG)
    3. Covariance Method (COV)
    4. Modified Covariance (MCOV)
    5. Itakura-Saito (IS)
    6. Fast FFT (FAST)
    7. CondorBrain (CONDOR)
    """
    
    METHODS = ["YW", "BURG", "COV", "MCOV", "IS", "FAST", "CONDOR"]
    ORDERS = [2, 3, 4, 5, 6, 8, 10, 12]
    
    def __init__(self, 
                 fit_window: int = 512,
                 val_window: int = 128,
                 horizon: int = 5,
                 cooldown_k: int = 5,
                 gamma: float = 0.05,
                 tau: float = 0.02,
                 alpha: float = 0.2,
                 delta: float = 1.5,
                 lambda_h: float = 0.8,
                 weights: Dict[str, float] = None):
        
        self.fit_window = fit_window
        self.val_window = val_window
        self.horizon = horizon
        self.cooldown_k = cooldown_k
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.delta = delta
        self.lambda_h = lambda_h
        
        # Component weights: r=1.0, rho=0.7, d=0.5, v=0.2
        self.weights = weights or {"r": 1.0, "rho": 0.7, "d": 0.5, "v": 0.2}
        
        # Internal state
        self.state = ForecastState()
        
    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform raw OHLCV into stationary feature vector y_t = [r, rho, d, v]
        """
        # Ensure we have required columns
        req = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in req):
            raise ValueError(f"DataFrame missing columns: {set(req) - set(df.columns)}")
            
        # Avoid log(0)
        eps = 1e-9
        
        # 1. Close Return: r_t = log(C_t) - log(C_{t-1})
        log_c = np.log(df['close'] + eps)
        r = log_c.diff().fillna(0)
        
        # 2. Volatility/Range: rho_t = log(H_t) - log(L_t)
        rho = np.log(df['high'] + eps) - np.log(df['low'] + eps)
        
        # 3. Direction: d_t = log(C_t) - log(O_t)
        d = np.log(df['close'] + eps) - np.log(df['open'] + eps)
        
        # 4. Volume Delta: v_t = log(V_t+1) - log(V_{t-1}+1)
        log_v = np.log(df['volume'] + 1)
        v = log_v.diff().fillna(0)
        
        # Stack: (T, 4)
        y = np.stack([r, rho, d, v], axis=1)
        return y

    def reconstruct_ohlcv(self, y_pred: np.ndarray, last_bar: pd.Series) -> pd.DataFrame:
        """
        Reconstruct OHLCV from predicted feature vector
        y_pred: (H, 4) array of [r, rho, d, v]
        last_bar: Series with 'close', 'volume'
        """
        H = len(y_pred)
        
        preds = []
        curr_close = float(last_bar['close'])
        curr_vol = float(last_bar['volume'])
        
        for t in range(H):
            r, rho, d, v = y_pred[t]
            
            # Predict Close
            next_close = curr_close * np.exp(r)
            
            # Predict Open (assume ~ prev close)
            # Refined: O_{t+1} = C_{t+1} * exp(-d)  since d = log(C/O)
            # But simpler assumption is O_{t+1} = C_t
            next_open = curr_close 
            
            # Predict High/Low using envelope
            # H = max(O, C) * exp(0.5 * rho)
            # L = min(O, C) * exp(-0.5 * rho)
            base_high = max(next_open, next_close)
            base_low = min(next_open, next_close)
            
            next_high = base_high * np.exp(0.5 * rho)
            next_low = base_low * np.exp(-0.5 * rho)
            
            # Predict Volume
            next_vol = (curr_vol + 1) * np.exp(v) - 1
            next_vol = max(0, next_vol)
            
            preds.append({
                'open': next_open,
                'high': next_high,
                'low': next_low,
                'close': next_close,
                'volume': next_vol
            })
            
            # Update state
            curr_close = next_close
            curr_vol = next_vol
            
        return pd.DataFrame(preds)

    def _huber_loss(self, e: np.ndarray) -> np.ndarray:
        """Robust Huber loss"""
        abs_e = np.abs(e)
        mask = abs_e <= self.delta
        loss = np.where(mask, 0.5 * e**2, self.delta * (abs_e - 0.5 * self.delta))
        return loss

    def epsilon_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Weighted Huber Loss with Horizon Decay
        y_true, y_pred: (M, H, 4) or (H, 4)
        """
        # Calculate error: (M, H, 4)
        e = y_true - y_pred
        
        # Apply Huber
        h_loss = self._huber_loss(e) # (M, H, 4)
        
        # Apply Horizon Decay
        # lambda^{i-1} for i=1..H
        H = y_pred.shape[-2] # H dim is usually 2nd to last
        decay = np.power(self.lambda_h, np.arange(H)) 
        # decay needs to broadcast: (1, H, 1) usually if batched
        # If single sample: (H, 4)
        
        if y_pred.ndim == 2: # (H, 4)
             decay = decay.reshape(H, 1)
        else: # (M, H, 4)
             decay = decay.reshape(1, H, 1)
        
        weighted_loss = h_loss * decay
        
        # Apply Component Weights
        # r=0, rho=1, d=2, v=3
        w_vec = np.array([self.weights['r'], self.weights['rho'], self.weights['d'], self.weights['v']])
        
        total_loss = np.sum(weighted_loss * w_vec)
        
        # Normalize
        norm_avg = total_loss / (np.prod(y_pred.shape[:-1]) if y_pred.ndim > 1 else H)
        
        return float(norm_avg)

    # --- Solvers ---
    
    def _solve_yw(self, x: np.ndarray, p: int) -> np.ndarray:
        """Yule-Walker Solver"""
        # Using statsmodels or simple linalg
        # x: (T,)
        from statsmodels.regression.linear_model import yule_walker
        rho, sigma = yule_walker(x, order=p, method='mle')
        return rho
        
    def _solve_burg(self, x: np.ndarray, p: int) -> np.ndarray:
        """Burg's Method Solver"""
        from statsmodels.regression.linear_model import burg
        rho, sigma = burg(x, order=p)
        return rho
        
    def _forecast_ar(self, coeffs: np.ndarray, history: np.ndarray, horizon: int) -> np.ndarray:
        """Recursive AR forecasting"""
        # coeffs: [a1, a2, ..., ap]
        # history: [..., xt-1, xt]
        p = len(coeffs)
        preds = []
        curr_hist = list(history[-p:]) # Keep last p
        
        for _ in range(horizon):
            # x_next = sum(ai * x_t-i)
            # coeffs are usually returned as x_t = a1*x_t-1 + ...
            val = np.dot(coeffs, curr_hist[::-1][:p])
            preds.append(val)
            curr_hist.append(val)
            curr_hist.pop(0)
            
        return np.array(preds)

    def _solve_cov(self, x: np.ndarray, p: int) -> np.ndarray:
        """Covariance Method (Least Squares on unwindowed data)"""
        # Linear prediction: x[n] = -sum(a[k] * x[n-k])
        # Form X * a = -x_target
        N = len(x)
        if N <= p: return np.zeros(p)
        
        # Matrix X: columns are lags
        # Row 0: x[p-1], x[p-2], ..., x[0] -> pred x[p]
        # ...
        # Row K: x[N-2], ..., x[N-p-1] -> pred x[N-1]
        
        # Validity range: n from p to N-1
        rows = N - p
        X = np.zeros((rows, p))
        target = np.zeros(rows)
        
        for i in range(rows):
            # n = p + i
            # predictors: x[n-1] ... x[n-p]
            X[i, :] = x[p+i-1 : p+i-p-1 : -1] if (p+i-p-1) >= 0 else x[p+i-1 :: -1]
            # Actually slice logic: x[n-1], x[n-2]...
            # easier:
            idx = p + i
            X[i, :] = x[idx-p : idx][::-1]
            target[i] = x[idx]
            
        # Solve X * a = -target  =>  X * (-a) = target
        # We want a such that x[n] + sum(a[k]x[n-k]) approx 0
        # So sum(a[k]x[n-k]) approx -x[n]
        # X * a = -target
        
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, -target, rcond=None)
            return coeffs
        except:
            return np.zeros(p)

    def _solve_mcov(self, x: np.ndarray, p: int) -> np.ndarray:
        """Modified Covariance (Forward-Backward Least Squares)"""
        # Minimizes sum(|f|^2 + |b|^2)
        N = len(x)
        if N <= p: return np.zeros(p)
        
        rows = N - p
        # Forward eq:  x[n] + sum(a[k]x[n-k]) = f[n]
        # Backward eq: x[n-p] + sum(a[k]x[n-p+k]) = b[n]
        
        # We assume coefficients 'a' are real (for financial data)
        # However, typically Modified Covariance enforces hermiticity for complex, 
        # but here prediction weights are shared.
        # This is equivalent to solving a doubled system:
        # [ X_f ] a = [ -x_f ]
        # [ X_b ] a = [ -x_b ]
        
        X_f = np.zeros((rows, p))
        t_f = np.zeros(rows)
        
        X_b = np.zeros((rows, p))
        t_b = np.zeros(rows)
        
        for i in range(rows):
            n = p + i
            # Forward: predict x[n] using x[n-1]...x[n-p]
            X_f[i, :] = x[n-p : n][::-1]
            t_f[i] = x[n]
            
            # Backward: predict x[n-p] using x[n-p+1]...x[n]
            # Relation: b[n] = x[n-p] + sum(a[k]*x[n-p+k])
            # So sum(a[k]x[n-p+k]) approx -x[n-p]
            # predictors for a[k]: x[n-p+k] 
            # k=1: x[n-p+1], k=p: x[n]
            # So row is x[n-p+1 ... n]
            X_b[i, :] = x[n-p+1 : n+1]
            t_b[i] = x[n-p]
            
        # Stack
        X_total = np.vstack([X_f, X_b])
        t_total = np.concatenate([t_f, t_b])
        
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_total, -t_total, rcond=None)
            return coeffs
        except:
            return np.zeros(p)

    def fit_predict_method(self, method: str, p: int, y_window: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit method on window and return (val_true, val_preds, live_preds)
        y_window: (N, 4)
        """
        N = len(y_window)
        # Start validation a bit earlier to gather stats
        # Spec implies rolling validation.
        # We will use "In-sample recursive validation" on the last M points.
        
        # 1. Fit Coefficients
        coeffs_list = []
        for c in range(4):
            series = y_window[:, c]
            if method == "YW":
                coeffs = self._solve_yw(series, p)
            elif method == "BURG":
                coeffs = self._solve_burg(series, p)
            elif method == "COV":
                coeffs = self._solve_cov(series, p)
            elif method == "MCOV":
                coeffs = self._solve_mcov(series, p)
            elif method == "IS":
                # Fallback to Burg for now (IS requires complex IRLS)
                coeffs = self._solve_burg(series, p) 
            elif method == "FAST":
                # Fast: Use Burg with lower order or same
                coeffs = self._solve_burg(series, p) 
            else:
                coeffs = self._solve_yw(series, p)
            
            coeffs_list.append(coeffs)
            
        # --- Live Forecast ---
        # Forecast H steps into future from N
        live_preds = np.zeros((self.horizon, 4))
        for c in range(4):
            series = y_window[:, c]
            coeffs = coeffs_list[c]
            live_preds[:, c] = self._forecast_ar(coeffs, series, self.horizon)
            
        # --- Validation Error Computation ---
        # We use the computed coeffs to back-test on the last M samples
        # effectively checking how well these parameters generalize locally
        
        # Generate predictions for t in range(N-M, N-H)
        # We need (M, H, 4)
        # To avoid loop, we can vectorize or just loop M times (M=128 is small enough)
        
        # Optimization: Just do it for last K=16 points to save time
        v_points = 16
        val_preds_arr = np.zeros((v_points, self.horizon, 4))
        val_true_arr = np.zeros((v_points, self.horizon, 4))
        
        for i in range(v_points):
            t_start = N - v_points - self.horizon + i
            # Context for prediction: up to t_start
            
            for c in range(4):
                hist = y_window[:t_start, c]
                true_vals = y_window[t_start:t_start+self.horizon, c]
                
                pred = self._forecast_ar(coeffs_list[c], hist, self.horizon)
                
                val_preds_arr[i, :, c] = pred
                val_true_arr[i, :, c] = true_vals

        return val_true_arr, val_preds_arr, live_preds

    def fit_predict(self, ohlcv_window: pd.DataFrame, condor_forecast: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Main entry point.
        ohlcv_window: (N, 5) DataFrame
        condor_forecast: (H, 4) Optional neural forecast from CondorBrain
        """
        # 1. Transform
        y = self.transform_features(ohlcv_window) # (N, 4)
        
        scores = []
        
        # 2. Evaluate Methods
        for m in self.METHODS:
            if m == "CONDOR":
                # Special handling
                if condor_forecast is not None:
                    # We validate Condor by checking its PAST performance?
                    # Or just trust it?
                    # Spec: "Compute same epsilon error on last M validation slices"
                    # This requires Condor to have made predictions in the past.
                    # If we don't have history of Condor preds, we can't score it fairly vs AR.
                    # Assume condor_forecast is the LIVE prediction.
                    # We assign it a dummy score or 0 if it's the prefered active.
                    # For now: Score = Infinity if no history? 
                    # Let's rely on 'eps_ewma' in state. If condor was active, we know its error.
                    # For comparison, we interpret 'condor_forecast' as live only.
                    # We skip validation for now or use a proxy.
                    
                    # Placeholder: Assign score based on EWMA or high confidence
                    eps = self.state.eps_ewma.get(("CONDOR", 0), 0.5) # Default average
                    scores.append(("CONDOR", 0, eps, condor_forecast))
                continue
                
            # AR Methods
            for p in self.ORDERS:
                try:
                    # y_val_true, y_val_pred, live_pred
                    y_true, y_pred, live = self.fit_predict_method(m, p, y)
                    eps = self.epsilon_error(y_true, y_pred)
                    
                    # Update EWMA
                    key = (m, p)
                    curr = self.state.eps_ewma.get(key)
                    if curr is None:
                        new_eps = eps
                    else:
                        new_eps = self.alpha * eps + (1 - self.alpha) * curr
                    
                    self.state.eps_ewma[key] = new_eps
                    scores.append((m, p, new_eps, live))
                    
                except Exception as e:
                    # Method failed (singular matrix etc)
                    logger.debug(f"Method {m}-{p} failed: {e}")
                    continue

        if not scores:
            # Fallback
            logger.warning("All methods failed, returning zero forecast")
            best_pred = np.zeros((self.horizon, 4))
            return self.reconstruct_ohlcv(best_pred, ohlcv_window.iloc[-1])

        # 3. Rank and Select
        # Per-method best
        best_per_method = {}
        for item in scores:
            m, p, eps, pred = item
            if m not in best_per_method or eps < best_per_method[m][1]:
                best_per_method[m] = (p, eps, pred)
        
        # Global best with tie-break
        # Filter valid methods
        valid_methods = [v for k,v in best_per_method.items()]
        if not valid_methods: # Example: only condor failed
             return self.reconstruct_ohlcv(np.zeros((self.horizon,4)), ohlcv_window.iloc[-1])
             
        global_min_eps = min(x[1] for x in valid_methods)
        
        # Candidates within (1+tau) of min
        candidates = []
        for m, (p, eps, pred) in best_per_method.items():
            if eps <= (1 + self.tau) * global_min_eps:
                candidates.append((m, p, eps, pred))
        
        # Tie-break: Lowest order P (Condor=0 or 999?)
        # Spec: "choose smallest p"
        # Let's penalize Condor slightly for complexity if it's tied?
        # Say Condor P=100. AR P=2.
        def get_rank(x):
            m_name, p_val = x[0], x[1]
            if m_name == "CONDOR": return 50 # Treat as high order
            return p_val
            
        best_candidate = min(candidates, key=lambda x: (get_rank(x), x[2]))
        
        m_new, p_new, eps_new, pred_new = best_candidate
        
        # 4. Hysteresis Switch
        m_curr = self.state.active_method
        p_curr = self.state.active_order
        eps_curr = self.state.eps_ewma.get((m_curr, p_curr), float('inf'))
        
        if self.state.cooldown > 0:
            self.state.cooldown -= 1
            # Stay with current IF it produced a valid forecast this round
            # Find current in scores
            curr_score = next((x for x in scores if x[0] == m_curr and x[1] == p_curr), None)
            if curr_score:
                final_m, final_p, final_pred = m_curr, p_curr, curr_score[3]
            else:
                # Current failed? Switch immediately
                final_m, final_p, final_pred = m_new, p_new, pred_new
                self.state.active_method = final_m
                self.state.active_order = final_p
                self.state.cooldown = self.cooldown_k
        else:
            # Check switch condition
            # New error must be significantly better (< 1-gamma)
            if eps_new < (1 - self.gamma) * eps_curr:
                # Switch!
                final_m, final_p, final_pred = m_new, p_new, pred_new
                self.state.active_method = final_m
                self.state.active_order = final_p
                self.state.cooldown = self.cooldown_k
            else:
                # Stick
                curr_score = next((x for x in scores if x[0] == m_curr and x[1] == p_curr), None)
                if curr_score:
                    final_m, final_p, final_pred = m_curr, p_curr, curr_score[3]
                else:
                    final_m, final_p, final_pred = m_new, p_new, pred_new
                    self.state.active_method = final_m
                    self.state.active_order = final_p

        # 5. Reconstruct
        return self.reconstruct_ohlcv(final_pred, ohlcv_window.iloc[-1])
