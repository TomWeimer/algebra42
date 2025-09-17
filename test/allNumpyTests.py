import unittest
import os
import sys
from datetime import datetime

import data.testData as testData
import data.testUtils as testUtils

# Get timestamp from command-line arguments
RUN_TIMESTAMP = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y%m%d_%H%M%S")

# Folders
TEST_FOLDER = "tests"
LOG_FOLDER = "logs"

LOG_FILE = "logs/data/numpy_data.log"

# Ensure logs folder exists
os.makedirs(LOG_FOLDER, exist_ok=True)

# Discover all test files recursively
test_files = []
test_logs = []

for root, dirs, files in os.walk(TEST_FOLDER):
    for f in files:
        if f.endswith(".py") and f != os.path.basename(__file__):  # exclude this runner
            full_test_path = os.path.join(root, f)
            test_files.append(full_test_path)

            # Log path mirrors folder structure
            rel_path = os.path.relpath(full_test_path, TEST_FOLDER)
            log_path = os.path.join(LOG_FOLDER, rel_path[:-3] + ".log")  # replace .py with .log
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            test_logs.append(log_path)

# Function to write log entries
def write_log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{msg}\n")

# Track overall success
all_passed = True

write_log(f"created at: [{RUN_TIMESTAMP}]")

# write test data
for name, data in testData.allData:
    write_log("\n" + name)
    testUtils.write_content(data, write_log)

    

for i, file in enumerate(test_files):
    logfile = test_logs[i]
    print(f"Running {file} -> {logfile}")

    try:
        
        LOG_FILE = logfile
        
        write_log(f"created at: [{RUN_TIMESTAMP}]")
         
        # Dynamically load the test module
        import importlib.util
    
        spec = importlib.util.spec_from_file_location(os.path.basename(file)[:-3], file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # If the test module defines a function like `run_tests(logfile)` you could call it:
        if hasattr(mod, "run_tests"):
            mod.run_tests(write_log)     
    except Exception as e:
        all_passed = False

# Final summary
if all_passed:
    print("All Python tests passed successfully!")
    sys.exit(0)
else:
    print("Some Python tests failed or errored!")
    sys.exit(1)