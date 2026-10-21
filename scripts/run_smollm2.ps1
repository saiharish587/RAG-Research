# Runs the SmolLM2 benchmark (2 models x 5 arms x 600 questions x 10 runs = 60,000 runs).
# Run from the repository root in PowerShell, with the virtual environment set up
# (see RUN_SMOLLM2.md). Safe to re-run: completed runs are skipped.

$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"
$csv = "results/csv/benchmark_results_smollm2.csv"
$models = @("hf.co/bartowski/SmolLM2-135M-Instruct-GGUF:Q4_K_M", "hf.co/bartowski/SmolLM2-360M-Instruct-GGUF:Q4_K_M")

# 1. Models
foreach ($m in $models) { ollama pull $m }

# 2. Record the environment the runs were produced on
$envFile = "results/smollm2_environment.txt"
"date: $(Get-Date -Format s)" | Out-File $envFile -Encoding utf8
"git commit: $(git rev-parse HEAD)" | Out-File $envFile -Append -Encoding utf8
"ollama: $(ollama --version)" | Out-File $envFile -Append -Encoding utf8
"python: $(& $python --version)" | Out-File $envFile -Append -Encoding utf8
nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv | Out-File $envFile -Append -Encoding utf8
foreach ($m in $models) { "model $m digest: $((ollama list | Select-String ([regex]::Escape($m))).ToString())" | Out-File $envFile -Append -Encoding utf8 }
& $python -m pip freeze | Out-File $envFile -Append -Encoding utf8

# 3. Preflight: retrieval must match the logged runs exactly
& $python scripts/check_retrieval.py 50
if ($LASTEXITCODE -ne 0) { throw "Retrieval differs from the logged runs. Stop and report this before running." }

# 4. Benchmark, one arm at a time across both models (Naive first)
foreach ($arm in @("naive", "oracle", "no_rag", "advanced", "modular")) {
    Write-Host "=== $(Get-Date -Format s) start $arm"
    & $python main.py --config configs/config_smollm2.yaml --output-csv $csv --arms $arm --skip-post *>&1 |
        Tee-Object -FilePath "results/smollm2_run_$arm.log"
    Write-Host "=== $(Get-Date -Format s) end $arm"
}
Write-Host "Done. Results: $csv and results/raw/*SmolLM2*.json"
