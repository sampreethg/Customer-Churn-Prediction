# PowerShell Launcher for Unified Multi-Domain AI Portfolio Hub
$ErrorActionPreference = "Stop"

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " Unified Multi-Domain AI Portfolio Hub: Launcher " -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

# Step 1: Pre-flight checks
Write-Host "`n[Step 1/5] Checking environment requirements..." -ForegroundColor Yellow

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is required but not installed."
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is required but not installed."
}

Write-Host "Core toolchain detected: Python, Docker, and Docker Compose." -ForegroundColor Green

# Step 2: Model Artifact Verification
Write-Host "`n[Step 2/5] Checking model artifacts..." -ForegroundColor Yellow

$modelsDir = "models"
$requiredModels = @(
    "$modelsDir/best_churn_model.joblib",
    "$modelsDir/best_cnn_model.keras",
    "$modelsDir/fine_tuned_distilbert/model.safetensors",
    "$modelsDir/best_per_d3qn.pt"
)

$missingArtifacts = $false
foreach ($m in $requiredModels) {
    if (-not (Test-Path $m)) {
        $missingArtifacts = $true
        break
    }
}

if ($missingArtifacts) {
    Write-Host "Artifacts missing or incomplete. Running baseline orchestrator..." -ForegroundColor Yellow
    python scripts/generate_baseline_artifacts.py
} else {
    Write-Host "All 4 model artifacts exist on disk." -ForegroundColor Green
}

# Step 3: Clean Existing Containers
Write-Host "`n[Step 3/5] Cleaning stale containers and networks..." -ForegroundColor Yellow
docker compose down --remove-orphans 2>$null | Out-Null

# Step 4: Build and Deploy the Multi-Container Stack
Write-Host "`n[Step 4/5] Building and launching containers in detached mode..." -ForegroundColor Yellow
docker compose up --build -d

# Step 5: Service Health Probe
Write-Host "`n[Step 5/5] Waiting for services to reach healthy status..." -ForegroundColor Yellow

$maxAttempts = 30
$attempt = 1
$apiHealthy = $false

while ($attempt -le $maxAttempts) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 2 -ErrorAction Stop
        if ($response.status -eq "healthy") {
            $apiHealthy = $true
            break
        }
    } catch {
        Write-Host -NoNewline "."
        Start-Sleep -Seconds 2
    }
    $attempt++
}
Write-Host ""

if ($apiHealthy) {
    Write-Host "FastAPI Gateway is online and healthy." -ForegroundColor Green
} else {
    Write-Host "FastAPI Gateway failed health check after 60 seconds." -ForegroundColor Red
    Write-Host "--- Container Logs ---" -ForegroundColor Yellow
    docker compose logs --tail=50 hub_api
    exit 1
}

Write-Host "`n======================================================" -ForegroundColor Green
Write-Host " Portfolio Hub is Live and Operational! " -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green
Write-Host "- Streamlit Dashboard: http://localhost:8501" -ForegroundColor Cyan
Write-Host "- FastAPI Swagger Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "- Gateway Health Check: http://localhost:8000/health" -ForegroundColor Cyan
Write-Host "`nTo view live container logs, run: docker compose logs -f" -ForegroundColor Yellow
Write-Host "To cleanly tear down the hub, run: docker compose down`n" -ForegroundColor Yellow

