#!/bin/bash
set -e


# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to that directory
cd "$SCRIPT_DIR" || exit 1

# Folders to clear
TARGET_FOLDER1="logs"

TARGET_FOLDER2="diffs"

# Python test script
NUMPY_TEST="allNumpyTests.py"

# Python test script
JULIA_TEST="allJuliaTests.jl"

# Python test script
COMPARE_LOGS="compareLog.jl"

readonly RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Safety check: make sure folder exists
if [ ! -d "$TARGET_FOLDER1" ]; then
    echo "Error: folder '$TARGET_FOLDER1' does not exist."
    exit 1
fi

if [ ! -d "$TARGET_FOLDER2" ]; then
    echo "Error: folder '$TARGET_FOLDER2' does not exist."
    exit 1
fi

echo "Run timestamp: $RUN_TIMESTAMP"

# Clear all files in the folder
echo "Clearing all files in $TARGET_FOLDER1..."
rm -rf "$TARGET_FOLDER1"/*

# Clear all files in the folder
echo "Clearing all files in $TARGET_FOLDER2..."
rm -rf "$TARGET_FOLDER2"/*

# Run the Julia test
echo "Running Julia test..."
julia --project=.. $JULIA_TEST $RUN_TIMESTAMP || { echo "Julia test failed"; exit 1; }

# Run the Python test
echo "Running Python test..."
python3 "$NUMPY_TEST" $RUN_TIMESTAMP > /dev/null 2>&1 || { echo "Python test failed"; exit 1; }

echo "Obtain Results..."
julia --project=.. -q -L $COMPARE_LOGS || { echo "test results failed"; exit 1; }

# Done
echo "Done."
