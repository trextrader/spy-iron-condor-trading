# Stage 1 Benchmarks

Device: cuda
GPU: Tesla T4
Torch: 2.4.0+cu121  CUDA: 12.1
Iterations: 100 (warmup=10)

## Selector (iron_condor)

| Strategy | K | S | CPU ms | GPU ms | Speedup |
|---|---|---|---|---|---|
| iron_condor | 8 | 50 | 0.828 | 2.078 | 0.40x |
| iron_condor | 32 | 50 | 1.925 | 1.790 | 1.07x |
| iron_condor | 128 | 50 | 8.465 | 1.805 | 4.69x |
| iron_condor | 512 | 50 | 39.144 | 1.815 | 21.57x |
| iron_condor | 8 | 200 | 0.503 | 1.795 | 0.28x |
| iron_condor | 32 | 200 | 1.954 | 1.805 | 1.08x |
| iron_condor | 128 | 200 | 11.063 | 2.454 | 4.51x |
| iron_condor | 512 | 200 | 40.102 | 2.746 | 14.60x |
| iron_condor | 8 | 500 | 1.047 | 2.656 | 0.39x |
| iron_condor | 32 | 500 | 3.575 | 2.551 | 1.40x |
| iron_condor | 128 | 500 | 9.107 | 1.809 | 5.03x |
| iron_condor | 512 | 500 | 47.470 | 1.830 | 25.94x |

## Mark-to-market (GPU only)

| K | S | GPU ms |
|---|---|---|
| 8 | 200 | 0.885 |
| 32 | 200 | 1.023 |
| 128 | 200 | 0.983 |
| 512 | 200 | 0.900 |
