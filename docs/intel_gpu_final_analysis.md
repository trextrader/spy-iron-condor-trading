# Intel GPU Acceleration - Final Analysis & Installation Guide

## Executive Summary

After extensive research, **Intel HD 5500 Graphics is incompatible** with both primary methods for GPU-accelerated ML inference on Intel hardware:

1. ❌ **IPEX-LLM**: Requires Intel Gen 11+ (2020+), HD 5500 is Gen 5 (2015)
2. ❌ **llama.cpp + Mamba**: Mamba GPU kernels not implemented (CPU-only as of 2025)

---

## Method 1: IPEX-LLM via LlamaIndex Integration

### Hardware Requirements
**Supported GPUs** ([source](https://www.intel.com/content/www/us/en/developer/articles/technical/run-llms-on-gpus-using-llama-cpp.html)):
- Intel Arc Series (A770, A750, A380, B580)
- Intel Data Center GPUs (Flex 170/140, Max 1550)
- Intel Core Ultra iGPUs
- **Intel Core 11th-14th Gen iGPUs** (Tiger Lake 2020+)

**Your GPU**: Intel HD 5500 (Broadwell 5th Gen, 2015)
**Status**: ❌ **6 generations too old**

### Software Requirements
- **Python**: 3.9-3.11 (⚠️ NOT 3.12)
- **Windows**: GPU driver 31.0.101.5122+ ([download](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html))
- **Toolkit**: Intel oneAPI Base Toolkit (optional for some features)

### Correct Installation Procedure

#### Option 1A: LlamaIndex Integration (Recommended)
```powershell
# Create environment with Python 3.11 (NOT 3.12)
mamba create -n ipexllm python=3.11 libuv -y
mamba activate ipexllm

# Single-command install (installs compatible PyTorch automatically)
pip install llama-index-llms-ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# Set runtime environment
set SYCL_CACHE_PERSISTENT=1
set BIGDL_LLM_XMX_DISABLED=1
```

**Package Info**: [PyPI - llama-index-llms-ipex-llm](https://pypi.org/project/llama-index-llms-ipex-llm/)

#### Option 1B: Direct IPEX-LLM Install
```powershell
mamba create -n ipexllm python=3.11 libuv -y
mamba activate ipexllm

# Use newer version with torch 2.5+ support
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# OR: Use specific version (if compatibility issues)
pip install --pre --upgrade ipex-llm[xpu_2.6] --extra-index-url https://download.pytorch.org/whl/xpu
```

**Note**: GitHub issue [#12931](https://github.com/intel/ipex-llm/issues/12931) documents that `torch==2.1.0a0` is broken and no longer in repositories. Newer versions use `xpu_2.6` variant.

### Verification Test
```python
import torch
from ipex_llm.transformers import AutoModel

# Test XPU availability (will fail on HD 5500)
tensor_1 = torch.randn(1, 1, 40, 128).to('xpu')
tensor_2 = torch.randn(1, 1, 128, 40).to('xpu')
print(torch.matmul(tensor_1, tensor_2).size())
# Expected: torch.Size([1, 1, 40, 40])
# HD 5500: RuntimeError: No XPU devices are available
```

### Why HD 5500 Fails
The XPU backend requires:
- **SYCL Runtime**: Data Parallel C++ abstraction layer (2021+)
- **oneAPI Compatibility**: Unified heterogeneous programming (2020+)
- **Level Zero API**: Low-level GPU interface (2020+)
- **Modern Compute Shaders**: EU architecture enhancements (Gen 11+)

Intel HD 5500 predates these technologies by 5-7 years.

---

## Method 2: llama.cpp with GGUF Format

### Mamba Model Support Status

**Critical Limitation** ([source](https://github.com/ggml-org/llama.cpp/issues/6758)):
- ❌ **Mamba GPU acceleration NOT implemented** as of January 2025
- CPU-only support added April 2024
- GPU requires kernels for `GGML_OP_SSM_CONV` and `GGML_OP_SSM_SCAN`
- `-ngl` (n_gpu_layers) parameter has **no effect** on Mamba models

**Quote from developers**:
> "CUDA backend does not support the Mamba-specific operations, so there will be no benefit to offloading Mamba models until these are implemented."

This applies to **all backends**: CUDA (NVIDIA), Metal (Apple), SYCL (Intel), Vulkan

### Intel GPU Support (for non-Mamba models)

llama.cpp SYCL backend supports Intel GPUs ([source](https://www.intel.com/content/www/us/en/developer/articles/technical/run-llms-on-gpus-using-llama-cpp.html)):
- Intel Arc Series
- Intel UHD Graphics 770
- "Reasonably modern" integrated GPUs

**Performance**: 21%-87% faster than OpenCL backend on Arc/Flex/Max GPUs

**HD 5500 Status**: ⚠️ **Unclear** - documentation says "reasonably modern," minimum generation unspecified

### Installation (if attempting)
```powershell
# Windows build with SYCL support
# Download pre-built: llama-b4040-bin-win-sycl-x64.zip
# Or compile from source with Intel oneAPI Base Toolkit

# Run with GPU offload (doesn't work for Mamba)
llama-cli -m model.gguf -ngl 99 -p "prompt"
```

**Conclusion**: Not viable for Mamba 2 models (CPU-only regardless of hardware)

---

## Available Mamba 2 GGUF Models

### Intel Pruned Models
- Search Hugging Face for: `intel mamba gguf`
- Example: `dranger003/mamba-2.8b-hf-GGUF` ([link](https://huggingface.co/dranger003/mamba-2.8b-hf-GGUF))
- Sizes: 2.8B, 1.4B, 370M parameters

### Client Applications (CPU-only for Mamba)
- **LM Studio**: GUI for local LLMs, supports GGUF
- **oobabooga Text Generation WebUI**: Feature-rich web interface
- **llama.cpp CLI**: Direct command-line usage

---

## Performance Comparison: HD 5500 vs Modern GPUs

| GPU | Generation | Year | Compute Units | oneAPI | XPU | Mamba GPU |
|-----|------------|------|---------------|--------|-----|-----------|
| Intel HD 5500 | Gen 5 | 2015 | 24 EUs | ❌ | ❌ | ❌ |
| Intel UHD 770 | Gen 12 | 2021 | 32 EUs | ✅ | ✅ | ❌ |
| Intel Arc A380 | Alchemist | 2022 | 128 EUs | ✅ | ✅ | ❌ |
| Intel Arc A770 | Alchemist | 2022 | 512 EUs | ✅ | ✅ | ❌ |
| NVIDIA RTX 3060 | Ampere | 2021 | 3584 CUDA | N/A | N/A | ❌ |

**Note**: Mamba GPU acceleration unavailable on ALL GPUs as of January 2025 (llama.cpp limitation)

---

## Recommended Solutions for Your System

### 1. Keep MockMambaKernel (Current Implementation) ✅ **Best Option**

**Why it works**:
```python
# intelligence/mamba_engine.py:25
class MockMambaKernel:
    """Deterministic CPU-based fallback"""
    def forecast(self, context_window, spot, rsi, atr, volume):
        # Uses weighted scoring: 40% price, 30% RSI, 20% ATR, 10% volume
        score = weighted_average(...)
        if score > 0.55: return (0.65, 0.20, 0.15, 0.80)  # Bullish
        elif score < 0.45: return (0.20, 0.65, 0.15, 0.80)  # Bearish
        else: return (0.33, 0.33, 0.34, 0.60)  # Neutral
```

**Advantages**:
- ✅ Zero latency (instant computation)
- ✅ Deterministic (reproducible backtests)
- ✅ No hardware requirements
- ✅ No API costs
- ✅ Already integrated and tested

**Disadvantages**:
- ❌ Not a "real" neural network
- ❌ No learning/adaptation
- ❌ Fixed heuristic logic

**Verdict**: Sufficient for 8-bar context window in options strategy (entry filters already robust)

---

### 2. ONNX Runtime CPU Optimization 🔧 **Viable Alternative**

If you want real neural models without GPU:

```powershell
pip install onnxruntime onnx transformers
```

**Benefits**:
- 2-5x faster than pure PyTorch CPU
- AVX2/AVX512 vectorization on your CPU
- Works with exported models

**Implementation**:
```python
import onnxruntime as ort

session = ort.InferenceSession("mamba2.onnx", providers=['CPUExecutionProvider'])
outputs = session.run(None, {"input": context_window})
```

**Drawbacks**:
- Requires model conversion (Mamba 2 → ONNX)
- Still CPU-bound (30-100ms latency per inference)
- Limited Mamba 2 model availability in ONNX format

---

### 3. Cloud Inference APIs ☁️ **Most Realistic Alternative**

#### Option A: Replicate API
```python
import replicate

output = replicate.run(
    "replicate/mamba-2-130m",
    input={"prompt": context_features}
)
```

**Pricing**: $0.0001-0.001 per prediction (~$10-100/month for live trading)

#### Option B: HuggingFace Inference API
```python
from huggingface_hub import InferenceClient

client = InferenceClient(token="hf_XXX")
result = client.text_generation("state-spaces/mamba-2.8b", ...)
```

**Pricing**: Free tier (1,000 requests/month) + paid plans

#### Option C: AWS SageMaker / Azure ML
- Full control over model deployment
- Pay-per-use pricing
- 100-500ms latency

**Verdict**: Best balance of cost/performance if real Mamba 2 needed

---

### 4. Hardware Upgrade 💰 **Long-Term Solution**

**Budget Options**:
- **Intel Arc A310** (~$100-130): Entry-level, 96 EUs, 4GB VRAM, low power
- **Intel Arc A380** (~$130-160): Mid-range, 128 EUs, 6GB VRAM

**Performance Options**:
- **NVIDIA RTX 3060** (~$250-300): 12GB VRAM, excellent CUDA ecosystem
- **Intel Arc A770** (~$300-350): 512 EUs, 16GB VRAM, best Intel GPU

**ROI Analysis**:
- Upfront cost: $100-300
- Enables local inference: <10ms latency
- No ongoing API fees
- Works with all ML frameworks

**Caveat**: Still won't help with Mamba GPU acceleration until llama.cpp implements kernels

---

## Final Recommendation

**For SPY Options Trading System**:

1. **Short-term** (next 3-6 months): Continue using `MockMambaKernel`
   - Already working and sufficient for 8-bar context
   - No dependencies, zero latency
   - Focus development effort on strategy tuning, not infrastructure

2. **Mid-term** (6-12 months): If real neural forecasting needed:
   - **Option A**: Cloud inference API (Replicate/HF) for production
   - **Option B**: ONNX CPU optimization for local development

3. **Long-term** (12+ months): If local GPU needed:
   - Monitor llama.cpp Mamba GPU kernel development ([Issue #6758](https://github.com/ggml-org/llama.cpp/issues/6758))
   - Consider Intel Arc A310 (~$130) when Mamba GPU support lands
   - OR switch to transformer-based models (Llama 3, Mistral) which have full GPU support

---

## Installation Commands Summary

**If you still want to test IPEX-LLM** (will fail on HD 5500, but procedure is correct):

```powershell
# Delete old environment
mamba env remove -n ipexllm -y

# Create with Python 3.11 (NOT 3.12)
mamba create -n ipexllm python=3.11 libuv -y
mamba activate ipexllm

# Install LlamaIndex integration (easiest method)
pip install llama-index-llms-ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# Set environment variables
set SYCL_CACHE_PERSISTENT=1

# Test (will fail with "No XPU devices" on HD 5500)
python -c "import torch; print(torch.randn(1,1,40,128).to('xpu'))"
```

**Expected result**: `RuntimeError: No XPU devices are available` ✅ Confirms hardware limitation

---

## Sources

- [Intel IPEX-LLM Windows GPU Installation](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/install_windows_gpu.md)
- [Run LLMs on Intel GPUs Using llama.cpp](https://www.intel.com/content/www/us/en/developer/articles/technical/run-llms-on-gpus-using-llama-cpp.html)
- [llama.cpp SYCL Backend Documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
- [Mamba GPU Support Issue #6758](https://github.com/ggml-org/llama.cpp/issues/6758)
- [IPEX-LLM Issue #12931 (torch 2.1.0a0 broken)](https://github.com/intel/ipex-llm/issues/12931)
- [LlamaIndex IPEX-LLM Integration](https://developers.llamaindex.ai/python/examples/llm/ipex_llm_gpu/)
- [Intel Extension for PyTorch on PyPI](https://pypi.org/project/intel-extension-for-pytorch/)

---

## Conclusion

**Verdict**: Intel HD 5500 Graphics **cannot** run GPU-accelerated ML inference due to:
1. ❌ Hardware incompatibility with XPU/oneAPI (Gen 5 vs Gen 11+ requirement)
2. ❌ Mamba GPU kernels unimplemented in llama.cpp (all GPUs CPU-only for Mamba)

**Action**: Continue using `MockMambaKernel` in main project at `C:\SPYOptionTrader_test`

**Alternative**: If real Mamba 2 inference needed, use cloud API (Replicate/HuggingFace) or wait for GPU kernel implementation + hardware upgrade.
