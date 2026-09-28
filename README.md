# LLM and GPU Benchmarks

Small Python scripts for experimenting with GPU inference throughput, `torch.compile`, batching, and device-to-device memory bandwidth.

## Scripts

- `bench_code.py` measures single-sequence generation throughput.
- `bench_compile.py` measures generation after compiling the model forward pass. The first generation call includes compilation warmup and is excluded from the timed run.
- `bench_batch.py` compares generation throughput across batch sizes. Large batches can require substantial GPU memory.
- `gpu_test.py` measures device-to-device copy bandwidth and allocates two 512 MiB CUDA buffers.

## Requirements

- Python and a CUDA-compatible PyTorch installation for your GPU and driver. Install PyTorch using the instructions at [pytorch.org](https://pytorch.org/get-started/locally/).
- Hugging Face Transformers for the language-model scripts:

  ```bash
  pip install transformers
  ```

- Internet access on the first model run to download `Qwen/Qwen2.5-0.5B` from Hugging Face.

## Run

Run a script from the project directory with the Python environment containing PyTorch and Transformers:

```bash
python bench_code.py
python bench_compile.py
python bench_batch.py
python gpu_test.py
```

Results depend on the GPU, driver, PyTorch/Transformers versions, and available memory. The scripts are exploratory benchmarks; compare results only under documented, consistent conditions.