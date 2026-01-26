
import torch
import gc
import sys

def clear_gpu_memory():
    """
    Clears GPU memory cache and runs garbage collection.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPU to clear.")
        return

    print(f"GPU Memory before clearing: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
    
    # 1. Force Garbage Collection of Python objects
    gc.collect()
    
    # 2. Clear PyTorch Cache
    torch.cuda.empty_cache()

    # 3. Optional: Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    
    print(f"GPU Memory after clearing:  {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")

if __name__ == "__main__":
    clear_gpu_memory()
