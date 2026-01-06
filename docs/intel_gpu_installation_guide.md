# Intel GPU (IPEX-LLM) Installation Guide for Mamba 2

## Hardware Compatibility Analysis

### Your System
- **GPU**: Intel HD Graphics 5500
- **Architecture**: Broadwell (5th Generation, 2015)
- **Status**: ⚠️ **NOT SUPPORTED**

### IPEX-LLM Requirements
According to [official Intel documentation](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/install_windows_gpu.md):

**Supported GPUs:**
- Intel Arc Series (A770, A750, A380, B580)
- Intel Core Ultra iGPUs
- Intel Core 11th-14th Gen iGPUs (2020+)
- Intel Flex Series (Flex 170, Flex 140)
- Intel Max Series (Max 1550)

**Minimum Requirement**: 11th Gen Intel Graphics (Tiger Lake, 2020) or newer

**Your GPU**: 5th Gen (Broadwell, 2015) - **6 generations too old**

---

## Installation Procedure (For Reference)

Even though HD 5500 is incompatible, here is the correct installation procedure for future reference or if you upgrade hardware:

### Step 1: Update GPU Drivers
Download driver version `31.0.101.5122` or later from:
https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html

**Reboot required after installation.**

### Step 2: Create Conda Environment
```powershell
# IMPORTANT: Use Python 3.11, not 3.12
mamba create -n ipexllm python=3.11 libuv -y
mamba activate ipexllm
```

### Step 3: Install IPEX-LLM
**Single command** (installs compatible PyTorch automatically):

```powershell
# For US region:
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# For China region:
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
```

⚠️ **Do NOT pre-install PyTorch separately** - IPEX-LLM will install the correct version

### Step 4: Set Runtime Environment Variables
```powershell
set SYCL_CACHE_PERSISTENT=1
```

### Step 5: Verify Installation
```python
import torch
from ipex_llm.transformers import AutoModel

# Test GPU availability
tensor_1 = torch.randn(1, 1, 40, 128).to('xpu')
tensor_2 = torch.randn(1, 1, 128, 40).to('xpu')
print(torch.matmul(tensor_1, tensor_2).size())
```

**Expected output**: `torch.Size([1, 1, 40, 40])`

---

## Your Installation Attempt Results

### What We Tried
1. ✅ Created conda environment with Python 3.12
2. ✅ Installed PyTorch 2.10.0+xpu with XPU dependencies
3. ❌ **Dependency conflict**: IPEX-LLM requires torch==2.1.0a0, but 2.10.0+xpu was installed
4. ❌ **Hardware incompatibility**: `RuntimeError: No XPU devices are available`

### Root Causes
1. **Wrong Python version**: Used 3.12 instead of required 3.11
2. **Wrong installation order**: Pre-installed PyTorch separately, causing version mismatch
3. **Hardware limitation**: Intel HD 5500 lacks XPU compute capabilities (Gen 5 vs Gen 11+ requirement)

---

## Recommended Alternatives

Since Intel HD 5500 cannot run XPU workloads, consider these options:

### Option 1: Keep MockMambaKernel (CPU) ✅ Recommended
**Pros:**
- Already implemented and working
- Sufficient for 8-bar context window
- No additional dependencies
- Zero hardware requirements

**Cons:**
- Not "real" neural network (fixed logic)
- No learning/adaptation

### Option 2: Cloud Inference APIs
**Services:**
- Replicate API: $0.0001-0.001 per prediction
- HuggingFace Inference API: Free tier + paid plans
- AWS SageMaker: Pay per use

**Pros:**
- Access to real models (Mamba 2, Llama 3, etc.)
- No local GPU needed
- Scalable

**Cons:**
- Latency (~200-500ms per request)
- Ongoing costs
- Internet dependency

### Option 3: Hardware Upgrade
**Budget Options:**
- Intel Arc A310: ~$100-130 (entry-level, low power)
- Intel Arc A380: ~$130-160 (better performance)
- NVIDIA RTX 3060 12GB: ~$250-300 (excellent CUDA support)

**Pros:**
- Full local control
- Low latency
- Works with all ML frameworks

**Cons:**
- Upfront cost
- Power requirements

### Option 4: ONNX CPU Optimization
Use ONNX Runtime with CPU optimizations:
```python
import onnxruntime as ort

# CPU with AVX2/AVX512 optimizations
session = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])
```

**Pros:**
- Works on current hardware
- 2-5x faster than pure PyTorch CPU
- No API costs

**Cons:**
- Still CPU-bound
- Requires model conversion

---

## Technical Deep Dive: Why HD 5500 Doesn't Work

### Architecture Comparison

| Feature | HD 5500 (2015) | Required (Gen 11+, 2020+) |
|---------|----------------|---------------------------|
| Compute Units | 24 EUs | 96+ EUs |
| Gen Graphics | Gen 5 | Gen 11+ |
| oneAPI Support | ❌ | ✅ |
| SYCL Runtime | ❌ | ✅ |
| DPC++ Support | ❌ | ✅ |
| XPU Backend | ❌ | ✅ |

### What's Missing
1. **SYCL Runtime Support**: Modern compute abstraction layer (similar to CUDA)
2. **oneAPI Compatibility**: Unified programming model for Intel hardware
3. **DPC++ Kernels**: Data Parallel C++ for heterogeneous computing
4. **XPU Device Class**: PyTorch's Intel GPU device type

The XPU backend was introduced circa 2021-2022 and requires hardware features not present in pre-2020 Intel GPUs.

---

## Conclusion

**Verdict**: Intel HD 5500 Graphics fundamentally cannot run IPEX-LLM workloads due to missing hardware capabilities required for Intel's modern XPU compute stack.

**Recommendation**: Continue using `MockMambaKernel` in the main project (`C:\SPYOptionTrader_test`). It provides sufficient functionality for the SPY options trading system's 8-bar context window without external dependencies.

If real neural forecasting is desired in the future, consider:
1. **Short-term**: ONNX CPU optimization or cloud inference APIs
2. **Long-term**: Hardware upgrade to Intel Arc A310+ or NVIDIA GPU

---

## References
- [IPEX-LLM Official Installation Guide](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/install_windows_gpu.md)
- [Intel Arc/Iris Xe Graphics Driver](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)
- [PyTorch XPU Extension Repository](https://pytorch-extension.intel.com/release-whl/stable/xpu/us/)
- [Intel Hardware Compatibility List](https://github.com/intel/ipex-llm?tab=readme-ov-file)
