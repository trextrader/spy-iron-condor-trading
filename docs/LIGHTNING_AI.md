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

## 2. Train Neural CDE Prototype
This uses the new self-contained CDE module (`intelligence/models/neural_cde.py`) which requires no extra libraries.

**Command:**
```bash
python intelligence/train_neural_cde.py \
    --data "data/processed/mamba_institutional_2024_1m_last 1mil_v21.csv" \
    --epochs 10 \
    --hidden 128 \
    --layers 2
```

**Expected Output:**
*   You should see a progress bar for `Training CDE`.
*   Loss should decrease.
*   Final model saved to: `models/neural_cde_proto.pth`.

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
