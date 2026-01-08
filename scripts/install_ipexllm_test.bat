@echo off
REM Intel GPU IPEX-LLM Installation Test Script
REM For Intel HD 5500 Graphics compatibility testing
REM Expected Result: Installation succeeds, but GPU detection fails

echo ========================================
echo Intel GPU IPEX-LLM Installation Test
echo ========================================
echo.
echo WARNING: This installation will likely fail GPU detection
echo Intel HD 5500 (Gen 5, 2015) is not supported
echo Minimum requirement: Intel Gen 11+ (2020+)
echo.
pause

echo.
echo Step 1: Removing old ipexllm environment (if exists)...
call mamba env remove -n ipexllm -y

echo.
echo Step 2: Creating new environment with Python 3.11...
echo (Python 3.12 is NOT supported - must use 3.11)
call mamba create -n ipexllm python=3.11 libuv -y

echo.
echo Step 3: Activating environment...
call mamba activate ipexllm

echo.
echo Step 4: Installing LlamaIndex IPEX-LLM integration...
echo This will automatically install compatible PyTorch XPU version
pip install llama-index-llms-ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

echo.
echo Step 5: Setting environment variables...
set SYCL_CACHE_PERSISTENT=1
set BIGDL_LLM_XMX_DISABLED=1

echo.
echo ========================================
echo Installation Complete
echo ========================================
echo.
echo Next Step: Test GPU availability
echo Run the following in Python:
echo.
echo     python -c "import torch; print(torch.randn(1,1,40,128).to('xpu'))"
echo.
echo Expected Result:
echo     RuntimeError: No XPU devices are available
echo.
echo This confirms Intel HD 5500 hardware limitation.
echo See docs\intel_gpu_final_analysis.md for alternatives.
echo.
pause
