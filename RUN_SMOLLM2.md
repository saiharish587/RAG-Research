# Running the SmolLM2 benchmark on another machine

These steps reproduce the setup used for the LFM2 runs and run the second model
family (SmolLM2-135M and SmolLM2-360M): 5 RAG arms, 600 questions, 10 runs each,
60,000 runs in total. Expect roughly a day of GPU time.

## Requirements

- Windows, NVIDIA GPU (RTX 3050 Laptop, 4 GB, was used), laptop **plugged in** and
  set to its best-performance power mode (the GPU must not be power-limited).
- Python 3.12, Git, and [Ollama](https://ollama.com/download) installed and running.

## 1. Get the code and data

```powershell
git clone https://github.com/saiharish587/RAG-Research.git
cd RAG-Research
```

The FAISS index, evaluation set and HotpotQA data are included in the repository.

## 2. Create the Python environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Check that PyTorch sees the GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

It must print `True`.

## 3. Run

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_smollm2.ps1
```

The script downloads both models, records the environment in
`results/smollm2_environment.txt`, checks that retrieval matches the original
runs exactly (it stops if not), and then runs the benchmark one arm at a time.
If it is interrupted, run the same command again: finished runs are skipped.

Do not change any other file and do not run `main.py` without
`--output-csv results/csv/benchmark_results_smollm2.csv`; that would write into
the original LFM2 results.

## 4. Send the results back

When the script prints `Done`, push only the result files:

```powershell
git add results/csv/benchmark_results_smollm2.csv results/smollm2_environment.txt results/raw/
git commit -m "Add SmolLM2 benchmark results"
git push origin main
```

`results/raw/` will contain about 60,000 new JSON logs whose names start with
`hf.co_bartowski_SmolLM2`. Pushing requires write access to the repository; if
you do not have it, zip `results/csv/benchmark_results_smollm2.csv`,
`results/smollm2_environment.txt` and the SmolLM2 files in `results/raw/` and
send them instead.
