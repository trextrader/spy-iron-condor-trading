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
        *   Go to **GitHub Settings > Developer Settings > Personal Access Tokens > Tokens (classic)**.
        *   Generate a new token with `repo` scope.
        *   Copy/Paste this token into the Colab prompt (input is hidden for security).
2.  **Install**: Run Cell 2 to install dependencies.
3.  **Optimize**: Run Cell 3 to begin the 5-Phase Optimization.
4.  **Save**: Run Cell 4 to copy results back to Drive.

## 💡 Troubleshooting
*   **"Mamba-SSM not found"**: Ensure you selected a **GPU Runtime**. The pre-built wheels for Mamba require CUDA.
*   **"Drive not mounted"**: Re-run Cell 1 and ensure you completed the pop-up authorization.
*   **Slow Uploads**: If you have massive CSVs (10GB+), consider uploading only the files you need for the backtest (e.g., `spy_options_intraday_large_with_greeks_m1.csv`) or using `gdown` if the file is hosted elsewhere.

---
**Why Cloud?**
*   **Speed**: Faster CPU cores for the backtest loop.
*   **GPU**: 10x-50x speedup for the Mamba Neural Network inference.
*   **Offloading**: Frees up your local PC.
