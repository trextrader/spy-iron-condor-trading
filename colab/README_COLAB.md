# ☁️ Running SPY Option Trader on Google Colab

This guide explains how to migrate your local project to Google Colab to leverage high-performance GPUs (T4/A100) for Mamba Neural Network training and generic optimization.

## 1. Prepare Google Drive
1.  **Zip your project**: Compress your entire `SPYOptionTrader_test` folder.
    *   *Important*: Ensure your `data/` folder (especially the large CSVs) is included.
2.  **Upload to Drive**:
    *   Go to [Google Drive](https://drive.google.com).
    *   Create a new folder named `SPYOptionTrader`.
    *   Upload/Unzip your project files into this folder.
    *   **Final Path Check**: You should see `My Drive/SPYOptionTrader/core/main.py`.

## 2. Open the Notebook
1.  Navigate to `colab/` folder inside your Drive.
2.  Right-click `spyt_cloud_optimizer.ipynb` -> Open with -> **Google Colaboratory**.

## 3. Configure Runtime
1.  In Colab, go to **Runtime** > **Change runtime type**.
2.  **Hardware accelerator**: Select **T4 GPU** (Standard) or **A100** (Pro).
3.  **High-RAM**: Recommended if you have >10GB of CSV data.

## 4. Execution
1.  **Mount Drive & Authenticate**: Run Cell 1.
    *   **GitHub Auth**: Since the repo is private, you will be prompted for credentials.
    *   **Username**: Your GitHub username.
    *   **Token**: A Personal Access Token (PAT).
        *   **Direct Link**: [https://github.com/settings/tokens](https://github.com/settings/tokens)
        *   Click **Generate new token (classic)**.
        *   Generate a new token with `repo` scope.
        *   Copy/Paste this token into the Colab prompt (input is hidden for security).
2.  **Install**: Run Cell 2 (or a new cell) with:
    ```python
    !pip install -r requirements.txt
    !pip install causal-conv1d>=1.4.0 mamba-ssm>=2.2.2
    ```
3.  **Optimize**: Run Cell 3 to begin the 5-Phase Optimization.
4.  **Save**: Run Cell 4 to copy results back to Drive.

## 💡 Troubleshooting
*   **"Mamba-SSM not found"**: Ensure you selected a **GPU Runtime**. The pre-built wheels for Mamba require CUDA.
*   **"Drive not mounted"**: Re-run Cell 1 and ensure you completed the pop-up authorization.
*   **Slow Uploads**: If you have massive CSVs (10GB+), consider uploading only the files you need for the backtest (e.g., `spy_options_intraday_large_with_greeks_m1.csv`) or using `gdown` if the file is hosted elsewhere.

### Performance Tips

1. **Memory Safety (OOM Protection)**
   - The optimizer uses **Chunked Loading** (`chunksize=500000`) and **Float32 Precision** to load massive datasets (2GB+) on standard Colab instances (12GB RAM).
   - Only data within the `--bt-start` and `--bt-end` range is kept in memory.

2. **Acceleration (15x Speedup)**
   - The Neural Engine doesn't need 1-minute resolution. The optimizer automatically **resamples spot data to 15-minute bars (`15T`)**.
   - This reduces simulation time from ~90s to ~10s per run without losing signal quality.

3. **Mamba Compilation**
   - Mamba requires `causal-conv1d` to be compiled against the specific CUDA version.
   - If you see `TypeError: NoneType object is not callable`, run:
     ```bash
     !pip uninstall -y causal-conv1d mamba-ssm
     !pip install causal-conv1d>=1.2.0 mamba-ssm --no-binary :all:
     ```

---
**Why Cloud?**
*   **Speed**: Faster CPU cores for the backtest loop.
*   **GPU**: 10x-50x speedup for the Mamba Neural Network inference.
*   **Offloading**: Frees up your local PC.


---

## Repository Sync Addendum (2026-01-24)

This document is part of the synchronized documentation set. The authoritative engineering spec and audit references are:

- `docs/INTEGRATION_PLAN_MASTER.md`
- `docs/INTERFACE_CATALOG.md`

Key alignment requirements:
1. Feature schema selection by **name** (V2.2) only; no CSV order dependence.
2. Dataset column order differs across years; schema validation must be strict.
3. Model config metadata (layers/heads/input_dim) must match deployed checkpoints.

If this document conflicts with the master spec, the master spec governs implementation.
