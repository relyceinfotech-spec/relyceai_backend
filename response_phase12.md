## CPU vs GPU Architecture Overview
- **CPU:** Latency-optimized, complex control flow.
- **GPU:** Throughput-optimized, many simple cores.
- **Focus:** General-purpose vs data-parallel workloads.

## CPU vs GPU Architecture Comparison Table
| Feature | CPU | GPU |
|---|---|---|
| Core design and count | Few high-performance, out-of-order cores | Hundreds to thousands of small, simpler cores |
| Execution model / threading | Coarse-grained threads with sophisticated scheduling | SIMT (single instruction, multiple threads) with warp scheduling |
| Memory hierarchy | Large caches (L1/L2/L3), strong latency sensitivity | High memory bandwidth, shared memory and fast on-die caches; coalesced access patterns |
| Programming model / ISA | General-purpose ISAs (x86-64, ARM); mature toolchains | Kernel-based models (CUDA/OpenCL); SIMT-friendly APIs |
| Typical workloads | Latency-sensitive, serial or lightly parallel tasks | Highly parallel workloads: graphics, ML, simulations |
| Power & scalability | Strong single-thread performance; scalable across many cores | High throughput, but power-intensive; scaling driven by parallelism |

## Practical Implications for System Design
- **Decision criterion:** Use CPU for latency-sensitive and control-flow heavy tasks.
- **Parallelism opportunity:** Use GPU for data-parallel workloads (matrix ops, rendering, inference).
- **Data movement:** Minimize CPU-GPU data transfers to avoid bottlenecks.
- **Tooling & deployment:** CPU toolchains are highly mature; GPU workflows often require specialized frameworks (CUDA/OpenCL) and careful kernel design.

## Related Questions
- How do memory hierarchies affect CPU vs GPU performance?
- When should you consider CPU-GPU heterogeneity in a system?
- What is SIMT, and how does it differ from SIMD?