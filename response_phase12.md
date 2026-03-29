Here's a concise, practical comparison of CPU and GPU architectures.

High-level takeaway
- CPUs: general-purpose, latency-focused, optimized for single-thread performance and complex control flow.
- GPUs: highly parallel, throughput-focused, optimized for data-parallel workloads with many lightweight threads.

Core design and parallelism
- CPU:
  - Few powerful cores (often 4 - 64 per socket, depending on class).
  - Deep pipelines, out-of-order execution, large caches.
  - Strong branch prediction and speculative execution.
- GPU:
  - Dozens to thousands of simpler cores organized into Streaming Multiprocessors (SMs) or compute units.
  - SIMT (Single Instruction, Multiple Threads) execution model.
  - Latency hides through massive thread-level concurrency and scheduling.

Execution model
- CPU:
  - Thread-centric, with sophisticated out-of-order engines; excels at branching, irregular workloads.
  - Excellent per-thread latency and cache-coherent sharing across cores.
- GPU:
  - Warp/Thread-block-centric; schedules thousands of threads to hide memory latency.
  - Divergence in a warp (different branches taken by different threads) can reduce efficiency.

Memory hierarchy and bandwidth
- CPU:
  - Multi-level caches (L1, L2, often L3); strong temporal locality and coherence across cores.
  - Moderate-to-high per-core bandwidth; emphasis on low-latency access.
- GPU:
  - Very high global memory bandwidth; several memory types (global, shared/local, constant, texture) with explicit management.
  - Shared memory (on-chip scratchpad) allows fast intra-block data sharing; memory access patterns (coalescing) greatly affect throughput.
  - Coherence is more limited across the entire device; synchronization is often explicit within kernels.

Vector units and instructions
- CPU:
  - Wide vector units (SSE/AVX-512, NEON, etc.) for data-level parallelism.
  - General-purpose instruction set; flexible handling of irregular workloads.
- GPU:
  - Strong emphasis on parallel throughput rather than general-purpose scalar performance.
  - Specialized units in modern GPUs (e.g., tensor cores for AI, RT cores for ray tracing) augment throughput for specific tasks.

Programming model
- CPU:
  - Traditional languages (C/C++, Rust, etc.) plus threading libraries; SIMD intrinsics and compiler auto-vectorization.
- GPU:
  - Kernel-based programming (CUDA, OpenCL, HIP, etc.) with explicit kernel launches and thread hierarchies.
  - Memory access patterns (coalescing, shared memory usage) and synchronization requirements are critical for performance.

Typical workloads
- CPU:
  - Latency-sensitive tasks, complex control flow, OS and system services, databases, single-threaded or lightly threaded workloads.
- GPU:
  - Data-parallel workloads: large-scale ML training/inference, graphics, physics simulations, numerical HPC tasks.

Energy, density, and thermals
- CPU:
  - Strong performance per watt for diverse workloads; tends to be more power-efficient on single-threaded or lightly threaded tasks.
- GPU:
  - High throughput at scale, but can be power-hungry; best utilized when workloads can exploit massive parallelism.

Convergence and system integration
- Many systems pair CPUs with GPUs (heterogeneous computing) to balance latency and throughput.
- Some architectures blur lines (APUs, integrated GPUs, unified memory models) but the fundamental tradeoffs remain: CPUs for general, GPUs for parallel throughput.

If you want, I can tailor this to a specific family (e.g., Intel x86 CPUs vs. NVIDIA GPUs, AMD GPUs, ARM CPUs) or to a concrete workload (ML inference, real-time graphics, scientific computing).