# Toy vLLM

An educational project to build a small vLLM-inspired inference engine step by step. The first stage is measuring the GPU and model behavior that future implementations can be compared against.

## Baseline Benchmarking

The initial GPU and Hugging Face generation experiments live in [`baseline_benchmarking/`](baseline_benchmarking/README.md). They cover single-sequence generation, `torch.compile`, batch throughput, and basic CUDA memory-copy bandwidth.

These results are hardware- and software-dependent; see the benchmark README for requirements and run instructions.