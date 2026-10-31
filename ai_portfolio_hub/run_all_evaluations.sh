#!/usr/bin/env bash
set -euo pipefail

# Color palette
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${CYAN}${BOLD}====================================================================${NC}"
echo -e "${CYAN}${BOLD}       AI Portfolio Hub: Master Verification & Benchmark Runner      ${NC}"
echo -e "${CYAN}${BOLD}====================================================================${NC}"

# 1. Dependency Validation & Fast Install
echo -e "\n${YELLOW}[Stage 1/5] Verifying Python ecosystem dependencies...${NC}"
PYTHON_CMD="python3"
if ! command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python"
fi

REQUIRED_PKGS=("httpx" "torch" "scipy" "pydantic")
MISSING_PKGS=()

for pkg in "${REQUIRED_PKGS[@]}"; do
    if ! $PYTHON_CMD -c "import $pkg" >/dev/null 2>&1; then
        MISSING_PKGS+=("$pkg")
    fi
done

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Installing missing packages: ${MISSING_PKGS[*]}...${NC}"
    pip install --quiet "${MISSING_PKGS[@]}"
    echo -e "${GREEN}✓ All evaluation dependencies installed.${NC}"
else
    echo -e "${GREEN}✓ All evaluation dependencies verified.${NC}"
fi

# 2. Check Gateway Readiness
echo -e "\n${YELLOW}[Stage 2/5] Probing FastAPI Gateway health...${NC}"
if ! curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo -e "${YELLOW}Gateway not responding on :8000. Launching hub stack via launch_hub.sh...${NC}"
    chmod +x launch_hub.sh
    ./launch_hub.sh
fi
echo -e "${GREEN}✓ Gateway online at http://localhost:8000${NC}"

# 3. Concurrent Multi-Service Smoke Test
echo -e "\n${YELLOW}[Stage 3/5] Executing asynchronous multi-endpoint smoke tests...${NC}"
$PYTHON_CMD scripts/smoke_test_gateway.py

# 4. Model Inference Acceleration & INT8 Quantization
echo -e "\n${YELLOW}[Stage 4/5] Executing ONNX Runtime conversion and INT8 benchmarking...${NC}"
$PYTHON_CMD scripts/optimize_onnx.py

# 5. MLOps Data Drift & Covariate Shift Audit
echo -e "\n${YELLOW}[Stage 5/5] Executing data drift and stability audit...${NC}"
$PYTHON_CMD mlops/drift_monitor.py

echo -e "\n${GREEN}${BOLD}====================================================================${NC}"
echo -e "${GREEN}${BOLD}       All System Pipelines Verified and Operational!               ${NC}"
echo -e "${GREEN}${BOLD}====================================================================${NC}"
echo -e "• Streamlit UI:        ${CYAN}http://localhost:8501${NC}"
echo -e "• Swagger OpenAPI:     ${CYAN}http://localhost:8000/docs${NC}"
echo -e "• Latest Drift Report: ${CYAN}$(ls -t artifacts/mlops/drift_report_*.html 2>/dev/null | head -n 1 || echo 'artifacts/mlops/')${NC}"
echo -e "• Optimized Weights:   ${CYAN}models/optimized/distilbert_int8.pt${NC}\n"
