#!/bin/bash
set -e

# ────────────────────────────────────────────────
# Colors
# ────────────────────────────────────────────────
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
BLUE='\033[1;34m'
CYAN='\033[1;36m'
RESET='\033[0m'

# ────────────────────────────────────────────────
# Setup
# ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

TARGET_FOLDERS=("logs" "diffs" "benchmark")

NUMPY_TEST="allNumpyTests.py"
JULIA_TEST="allJuliaTests.jl"
COMPARE_LOGS="compareLog.jl"
COMPARE_BENCH="compareBenchmark.jl"

RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse command-line flags
BENCH_FLAG=""
if [[ "$1" == "--benchmark" ]]; then
    BENCH_FLAG="--benchmark"
    echo -e "${YELLOW}🏎️  Benchmark mode enabled${RESET}\n"
fi

# ────────────────────────────────────────────────
# Clear folders
# ────────────────────────────────────────────────
echo -e "${CYAN}🧹 Clearing old test outputs...${RESET}"

for folder in "${TARGET_FOLDERS[@]}"; do
    if [ -d "$folder" ]; then
        echo -e "  ${CYAN}→ Clearing ${YELLOW}$folder${RESET}"
        rm -rf "$folder"/*
    else
        echo -e "  ${YELLOW}⚙️  Creating missing folder ${folder}${RESET}"
        mkdir -p "$folder"
    fi
done

# ────────────────────────────────────────────────
# Run tests
# ────────────────────────────────────────────────
echo -e "\n${CYAN}🚀 Running Julia tests...${RESET}"
julia --project=.. "$JULIA_TEST" "$RUN_TIMESTAMP" "$BENCH_FLAG" || { echo -e "${RED}Julia test failed${RESET}"; exit 1; }

echo -e "\n${CYAN}🐍 Running Python tests...${RESET}"
python3 "$NUMPY_TEST" "$RUN_TIMESTAMP" > /dev/null 2>&1 || { echo -e "${RED}Python test failed${RESET}"; exit 1; }

echo -e "\n${CYAN}📊 Comparing results...${RESET}"
julia --project=.. -q -L "$COMPARE_LOGS" || { echo -e "${RED}Comparison failed${RESET}"; exit 1; }

# ────────────────────────────────────────────────
# Done
# ────────────────────────────────────────────────
echo -e "\n${GREEN}✅ All tests completed successfully!${RESET}"


if [[ "$BENCH_FLAG" == "--benchmark" ]]; then
    julia --project=.. -q -L "$COMPARE_BENCH" || { echo -e "${RED}Comparison failed${RESET}"; exit 1; }
fi