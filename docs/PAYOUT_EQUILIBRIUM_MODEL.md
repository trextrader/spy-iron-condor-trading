# PAYOUT EQUILIBRIUM MODEL: COMPLETE MULTI TIER DOCUMENTATION

## 1. Purpose and High Level Overview

### 1.1 High School Student Version
This tool shows where the market “wants” the price to go based on how many options traders are betting on different price levels. It finds the price where the total money the market would lose is the smallest. That price is called the equilibrium.

The chart then shows:
- Zones where the market is more likely to go up or down
- How much pressure options traders put on the price
- Where big changes in price behavior might happen
- A heatmap showing where payouts are high or low
- A table summarizing everything in simple numbers

It’s like seeing the “gravity map” of the market.

### 1.2 Professional Options Trader Version
This indicator computes the minimum payout equilibrium across a synthetic strike universe, modeling dealer inventory, gamma exposure, put/call OI asymmetry, and time to expiry decay. It identifies the price level where aggregate option seller payout is minimized, which often aligns with pinning, gamma neutrality, and dealer hedging flows.

It then visualizes:
- Tight and full range bullish/bearish premium zones
- Gamma flip levels
- Dealer net gamma
- Payout heatmap
- Strategy markers (`short_call`, `long_put`, `short_put`, `long_call`)
- Major strike levels
- A full equilibrium summary panel

This is a dealer flow driven price map.

### 1.3 Scientist / Mathematician Version
The model constructs a discrete approximation of the option payout functional:

$$ P(x) = \sum_{i=1}^N \left[ C_i \cdot \max(K_i - x, 0) + P_i \cdot \max(x - K_i, 0) \right] $$

where:
- $x$ is the hypothetical settlement price
- $K_i$ are synthetic strikes
- $C_i$, $P_i$ are modeled call/put OI
- Gamma exposure is approximated via a simplified Black Scholes gamma kernel

The equilibrium is:

$$ x^* = \arg\min_x P(x) $$

This is the global minimizer of the payout functional.

The indicator then constructs:
- A compressed modeling band
- A separate compressed visual band
- Soft clamped full range zones
- A payout heatmap via normalized gradient mapping
- A gamma exposure profile via normalized bar lengths
- A summary table

This is a functional minimization + flow field visualization of the options market.

## 2. Strike Universe and Synthetic Market Construction

### 2.1 High School Version
The script creates a list of possible prices (“strikes”) above and below the current price. It uses these to estimate how many traders are betting up or down.

### 2.2 Professional Options Trader Version
A synthetic strike grid is generated using:
- Scaled strike count based on timeframe
- Compressed strike range (0.25×)
- Round strike logic
- Synthetic OI estimation using distance weighted decay
- Put/call ratio adjustments
- Weekly options multiplier

This produces a dealer like strike distribution.

### 2.3 Scientist Version
The strike universe is:

$$ K_i = K_{\min} + i \cdot \Delta K $$

with:

$$ \Delta K = \max\left(\text{strike\_increment}, \frac{K_{\max} - K_{\min}}{N}\right) $$

OI is modeled as:

$$ OI(K) = V_0 \cdot f(d) \cdot g(d) $$

where:
- $d = \frac{|K - S|}{S}$
- $f(d) = \frac{1}{1 + 8d}$
- $g(d) = 0.8^{15d}$

Gamma exposure:

$$ \Gamma(K) = (C(K) - P(K)) \cdot \gamma_{BS}(S, K, T, \sigma) $$

## 3. Equilibrium Price (Minimum Payout)

### 3.1 High School Version
The equilibrium price is where the total money the market would lose is the smallest. This is where price often gets “pulled” toward.

### 3.2 Professional Options Trader Version
The equilibrium is the minimum aggregate payout across the synthetic OI surface. This often aligns with:
- Dealer hedging neutrality
- Gamma pinning
- Expiry driven magnet levels
- Low volatility compression zones

### 3.3 Scientist Version
We compute:

$$ x^* = \arg\min_x P(x) $$

via discrete evaluation across the strike grid.
This is a global minimization of a piecewise linear convex functional.

## 4. Band Ranges (Modeling vs Visual)

### 4.1 High School Version
Two “bands” are drawn around the equilibrium:
- One for calculations
- One for visuals

Both are compressed to keep the chart readable.

### 4.2 Professional Options Trader Version
You use:
- Modeling band: compressed 0.25×
- Visual band: compressed 0.25× (separate variable)

This allows:
- Mathematical purity
- Visual clarity
- Future independent tuning

### 4.3 Scientist Version
Modeling band:

$$ B_m = S \cdot \frac{p}{100} \cdot \sqrt{\Delta t} \cdot 0.25 $$

Visual band:

$$ B_v = S \cdot \frac{p}{100} \cdot \sqrt{\Delta t} \cdot 0.25 $$

Soft clamping:

$$ \text{clamp}(x) = \min(\max(x, L - 0.5R), U + 0.5R) $$

## 5. Zone Definitions

### 5.1 High School Version
The chart shows:
- Bearish zones (price likely to fall)
- Bullish zones (price likely to rise)
- Tight zones (stronger signals)
- Full range zones (weaker but broader signals)

### 5.2 Professional Options Trader Version
Zones correspond to premium pressure:
- **Tight Bearish Premium Zone**: short_call + long_put
- **Full Range Bearish Zone**: macro short_call + long_put
- **Equilibrium Band**: short_call + short_put (short vol)
- **Full Range Bullish Zone**: macro short_put + long_call
- **Tight Bullish Premium Zone**: short_put + long_call

These reflect dealer inventory pressure and volatility surface curvature.

### 5.3 Scientist Version
Zones are constructed via:

$$ Z_{\text{tight}} = [x^* \pm B_v] $$
$$ Z_{\text{full}} = [x^* \pm 3B_v] $$

Soft clamped to chart bounds.

## 6. Gamma Exposure

### 6.1 High School Version
Gamma shows how much the market “pushes back” when price moves. High gamma = price moves less. Low gamma = price moves more.

### 6.2 Professional Options Trader Version
Gamma bars show:
- Dealer long gamma (green)
- Dealer short gamma (red)
- Net gamma bias
- Gamma flip level

This is the hedging pressure map.

### 6.3 Scientist Version
Gamma exposure:

$$ \Gamma_{\text{exp}}(K) = (C(K) - P(K)) \cdot \gamma_{BS}(S, K, T, \sigma) $$

Gamma flip is the root of:

$$ \Gamma_{\text{exp}}(K) = 0 $$

## 7. Payout Heatmap

### 7.1 High School Version
Shows where payouts are high (red) or low (blue). The equilibrium is the lowest payout area.

### 7.2 Professional Options Trader Version
This is a normalized payout gradient across synthetic settlement prices. It highlights:
- Pin zones
- High risk wings
- Dealer stress points

### 7.3 Scientist Version
Heatmap uses:

$$ H(x) = 1 - \frac{P(x) - P_{\min}}{P_{\max} - P_{\min}} $$

Mapped to a color gradient.

## 8. Summary Table

### 8.1 High School Version
Shows:
- Current price
- Equilibrium price
- Distance to equilibrium
- Days to expiry
- Strike count
- Pinning bias
- Model mode

### 8.2 Professional Options Trader Version
Pinning bias is derived from:

$$ \left| \frac{x^* - S}{S} \right| $$

Strike count reflects synthetic grid density. Model regime reflects computational complexity.

### 8.3 Scientist Version
Distance:

$$ d = \frac{x^* - S}{S} \cdot 100 $$

Pinning classification:
- $|d| < 2\%$ → HIGH
- $|d| < 5\%$ → MEDIUM
- else → LOW

## 9. Integration Into CondorNet v4.6

### 9.1 High School Version
Your AI can use these zones and numbers to learn how the market behaves around options expiration.

### 9.2 Professional Options Trader Version
CondorNet can ingest:
- Equilibrium level
- Distance to equilibrium
- Net gamma
- Gamma flip
- Heatmap minima
- Zone boundaries
- Synthetic OI distributions
- Strike grid curvature

These become regime features for:
- Volatility forecasting
- Directional bias
- Pinning probability
- Dealer flow modeling

### 9.3 Scientist Version
CondorNet can treat the equilibrium model as a feature manifold:

$$ \mathcal{F} = \{x^*, d, \Gamma_{\text{net}}, \Gamma_{\text{flip}}, H(x), Z_{\text{tight}}, Z_{\text{full}}, OI(K), \Gamma(K)\} $$

This becomes a structured input tensor for:
- Regime classification
- Flow field prediction
- Stability analysis
- Multi horizon forecasting

The equilibrium functional $P(x)$ becomes a latent supervisory signal.
