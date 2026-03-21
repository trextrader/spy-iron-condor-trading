# Stage 3A Compile Benchmark

GPU: Tesla T4  Torch: 2.5.1+cu121  CUDA: 12.1
T_default: 2000  Iters: 3 (best-of steady-state)

| Mode | K | T | wall_s | bars/s | vs_Stage2A | trades |
|---|---|---|---|---|---|---|
| stage2a_gpu_eager | 32 | 500 | 0.180 | 2784 | 1.00x | 0 |
| step_bar_eager | 32 | 500 | 0.922 | 542 | 0.20x | 0 |
| step_bar_compile_warmup _includes JIT compilation cost_ | 32 | 500 | 5.953 | 84 | 0.03x | 0 |
| step_bar_compile_steady _steady-state (post-warmup)_ | 32 | 500 | 0.542 | 922 | 0.33x | 0 |
| stage2a_gpu_eager | 128 | 500 | 0.181 | 2769 | 1.00x | 0 |
| step_bar_eager | 128 | 500 | 0.919 | 544 | 0.20x | 0 |
| step_bar_compile_warmup _includes JIT compilation cost_ | 128 | 500 | 0.567 | 882 | 0.32x | 0 |
| step_bar_compile_steady _steady-state (post-warmup)_ | 128 | 500 | 0.541 | 924 | 0.33x | 0 |
| stage2a_gpu_eager | 32 | 2000 | 3.870 | 517 | 1.00x | 32 |
| step_bar_eager | 32 | 2000 | 5.345 | 374 | 0.72x | 32 |
| step_bar_compile_warmup _includes JIT compilation cost_ | 32 | 2000 | 3.808 | 525 | 1.02x | 32 |
| step_bar_compile_steady _steady-state (post-warmup)_ | 32 | 2000 | 3.818 | 524 | 1.01x | 32 |
| stage2a_gpu_eager | 128 | 2000 | 3.926 | 510 | 1.00x | 128 |
| step_bar_eager | 128 | 2000 | 5.328 | 375 | 0.74x | 128 |
| step_bar_compile_warmup _includes JIT compilation cost_ | 128 | 2000 | 3.895 | 514 | 1.01x | 128 |
| step_bar_compile_steady _steady-state (post-warmup)_ | 128 | 2000 | 3.812 | 525 | 1.03x | 128 |
