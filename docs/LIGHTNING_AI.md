# ⚡ Lightning AI / Cloud Processing Instructions

Follow these steps to run the new **Neural CDE** training and **Rule-Based Validation** on a Lightning AI Studio (or any cloud GPU environment).

## 1. Setup Environment
First, ensure you are in the project root:
```bash
cd spy-iron-condor-trading
# Or whatever the repo folder is named
```

Install dependencies if needed (Standard PyTorch is sufficient for the CDE prototype):
```bash
pip install torch pandas numpy tqdm
```

**Command:**
```bash
python intelligence/train_condor_brain.py \
    --local-data "data/processed/mamba_institutional_2024_1m_last 1mil_v21.csv" \
    --cde \
    --epochs 20 \
    --d-model 512 \
    --n-layers 3
```

**Expected Output:**
*   You should see a progress bar for `Training CondorBrain (CDE Mode)`.
*   Loss should decrease (BF16 precision).
*   Final model saved to: `models/condor_brain_cde_final.pth`.

## 3. Run Rule-Based Backtest (Baseline)
While the model trains, verify the P&L pipeline using the **Rule Engine** alone. This bypasses the neural network and makes trades based strictly on your heuristic rules (RSI, VIX, etc.).

**Command:**
```bash
python kaggle/condor_brain_backtest_v2.py \
    --input "data/processed/mamba_institutional_2024_1m_last 1mil_v21.csv" \
    --ruleset "docs/Complete_Ruleset_DSL.yaml" \
    --rules-only
```

**What to look for:**
*   "🛠️ RULES-ONLY MODE: Initializing Dummy CondorBrain..."
*   Trades being entered based on "Rule Signal".
*   Final P&L/Sharpe reported at the end.

## 4. (Optional) Run TFT Transformer
If you want to try the Transformer model, you must install the forecasting library first:

```bash
pip install pytorch-forecasting pytorch-lightning
```

Then run:
```bash
python kaggle/condor_brain_retrain_tft.py \
    --input "data/processed/mamba_institutional_2024_1m_last 1mil_v21.csv" \
    --max-epochs 10
```
