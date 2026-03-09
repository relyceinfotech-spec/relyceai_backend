## Overview
- **CPU:** General-purpose cores with strong single-thread performance.
- **GPU:** Thousands of simple cores targeting data-parallel throughput.
- **Decision:** Pick CPU vs GPU based on parallelism, memory needs, and tooling.

## Comparison Table
| Attribute | CPU | GPU |
|---|---|---|
| Architecture | Few, powerful cores with rich control flow | Many simple cores; SIMT/SIMD execution |
| Parallelism | Optimized for low-latency, irregular workloads | Massive data-parallel throughput |
| Memory System | Deep cache hierarchy; moderate per-core bandwidth | High bandwidth; large, parallel memory access |
| Programming Model | Thread-based, imperative; rich libraries | Data-parallel kernels; CUDA/OpenCL/ROCm; SIMT |
| Typical Workloads | OS, databases, interactive apps, complex logic | Graphics, ML, simulations, video encoding |
| Strengths | Flexibility; low-latency handling of diverse tasks | Throughput; parallelism; bandwidth utilization |

## Practical Guidance
- CPU: Use for latency-sensitive tasks with irregular control flow and small-to-medium data.
- GPU: Use for data-parallel workloads with large datasets and high throughput needs.
- Hybrid patterns: Orchestrate with CPU and offload compute to GPU when appropriate.
- Consider tooling, ecosystem maturity, and energy constraints before committing to a path.

## Related Questions
- Clarify your primary workload type (ML, graphics, analytics).
- Would you like a quick CPU-to-GPU migration checklist?
- What bottlenecks worry you most: latency, memory, or tooling?