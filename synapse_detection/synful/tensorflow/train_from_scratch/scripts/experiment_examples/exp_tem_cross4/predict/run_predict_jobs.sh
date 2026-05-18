#!/bin/bash

# Define the Python script to be executed
PYTHON_SCRIPT="04_predict_extract_blockwise.py"

# Define the list of JSON parameter files
# Make sure these files are in the same directory as this bash script,
# or provide the full path to them.
JSON_FILES=(
    "predict_extract_parameters_cremi_a.json"
    "predict_extract_parameters_cremi_b.json"
    "predict_extract_parameters_cremi_c.json"
    )

# Set CUDA_VISIBLE_DEVICES if required.
# If you need a different device for each run, you'll need to modify this loop.
# For now, it's set globally to '3' as per your example.
export CUDA_VISIBLE_DEVICES=0

echo "Starting sequential prediction process..."

# Loop through each JSON file and execute the Python script
for json_file in "${JSON_FILES[@]}"; do
    echo "----------------------------------------------------"
    echo "Processing: $json_file"
    echo "Running command: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $PYTHON_SCRIPT $json_file"

    # Execute the Python script. The script will wait for this command to finish
    # before moving to the next iteration of the loop.
    python "$PYTHON_SCRIPT" "$json_file"

    # Check the exit status of the last command
    if [ $? -eq 0 ]; then
        echo "Successfully completed: $json_file"
    else
        echo "Error: Python script failed for $json_file"
        # You can add error handling here, e.g., exit the script or log the error
        # exit 1 # Uncomment to stop the script on the first error
    fi
done

echo "----------------------------------------------------"
echo "All prediction jobs completed."

