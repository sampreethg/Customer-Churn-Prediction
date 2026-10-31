#!/usr/bin/env bash
set -euo pipefail

# Visual Formatting
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}   Unified Multi-Domain AI Portfolio Hub: Launcher   ${NC}"
echo -e "${BLUE}======================================================${NC}"

# Step 1: Pre-flight System & Dependency Checks
echo -e "\n${YELLOW}[Step 1/5] Checking environment requirements...${NC}"

command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1 || { echo -e "${RED}Python is required but not installed.${NC}"; exit 1; }
PYTHON_CMD="python3"
if ! command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python"
fi
command -v docker >/dev/null 2>&1 || { echo -e "${RED}Docker is required but not installed.${NC}"; exit 1; }
docker compose version >/dev/null 2>&1 || { echo -e "${RED}Docker Compose v2 is required but not installed.${NC}"; exit 1; }

echo -e "${GREEN}✓ Core toolchain detected: Python, Docker, and Docker Compose.${NC}"

# Step 2: Model Artifact Verification
echo -e "\n${YELLOW}[Step 2/5] Checking model artifacts...${NC}"

MODELS_DIR="models"
REQUIRED_MODELS=(
    "${MODELS_DIR}/best_churn_model.joblib"
    "${MODELS_DIR}/best_cnn_model.keras"
    "${MODELS_DIR}/fine_tuned_distilbert/model.safetensors"
    "${MODELS_DIR}/best_per_d3qn.pt"
)

MISSING_ARTIFACTS=0
for model in "${REQUIRED_MODELS[@]}"; do
    if [ ! -f "$model" ]; then
        MISSING_ARTIFACTS=1
        break
    fi
done

if [ $MISSING_ARTIFACTS -eq 1 ]; then
    echo -e "${YELLOW}Artifacts missing or incomplete. Running baseline orchestrator...${NC}"
    $PYTHON_CMD scripts/generate_baseline_artifacts.py
else
    echo -e "${GREEN}✓ All 4 model artifacts exist on disk.${NC}"
fi

# Step 3: Clean Existing Containers
echo -e "\n${YELLOW}[Step 3/5] Cleaning stale containers and networks...${NC}"
docker compose down --remove-orphans >/dev/null 2>&1 || true

# Step 4: Build and Deploy the Multi-Container Stack
echo -e "\n${YELLOW}[Step 4/5] Building and launching containers in detached mode...${NC}"
docker compose up --build -d

# Step 5: Service Health Probe
echo -e "\n${YELLOW}[Step 5/5] Waiting for services to reach healthy status...${NC}"

MAX_ATTEMPTS=30
ATTEMPT=1
API_HEALTHY=0

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:8000/health | grep -q '"status":"healthy"' 2>/dev/null; then
        API_HEALTHY=1
        break
    fi
    echo -n "."
    sleep 2
    ATTEMPT=$((ATTEMPT + 1))
done
echo ""

if [ $API_HEALTHY -eq 1 ]; then
    echo -e "${GREEN}✓ FastAPI Gateway is online and healthy.${NC}"
else
    echo -e "${RED}✗ FastAPI Gateway failed health check after 60 seconds.${NC}"
    echo -e "${YELLOW}--- Container Logs ---${NC}"
    docker compose logs --tail=50 hub_api
    exit 1
fi

echo -e "\n${GREEN}======================================================${NC}"
echo -e "${GREEN}   Portfolio Hub is Live and Operational!            ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo -e "• Streamlit Dashboard:    ${BLUE}http://localhost:8501${NC}"
echo -e "• FastAPI Swagger Docs:   ${BLUE}http://localhost:8000/docs${NC}"
echo -e "• Gateway Health Check:   ${BLUE}http://localhost:8000/health${NC}"
echo -e "\nTo view live container logs, run:   ${YELLOW}docker compose logs -f${NC}"
echo -e "To cleanly tear down the hub, run:  ${YELLOW}docker compose down${NC}\n"
