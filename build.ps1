# Build script for CUDA Genetic Algorithm
# Run from the directory containing your source files

$OutputName = "cuda_test"

Write-Host "Building $OutputName..." -ForegroundColor Cyan

nvcc -allow-unsupported-compiler -o $OutputName main.cpp ga_cpu.cpp ga_runner.cu cuda_kernels.cu -lcurand

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful! Run with: .\$OutputName.exe" -ForegroundColor Green
} else {
    Write-Host "Build failed." -ForegroundColor Red
}
