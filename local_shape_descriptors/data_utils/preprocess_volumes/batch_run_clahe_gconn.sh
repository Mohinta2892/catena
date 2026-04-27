#!/bin/bash
# run_clahe_on_folder.sh
# Usage: bash run_clahe_on_folder.sh /path/to/zarr/folder [kernel_z] [kernel_y] [kernel_x]

FOLDER="${1:?Please provide a folder path as the first argument}"
KERNEL_Z="${2:-30}"
KERNEL_Y="${3:-120}"
KERNEL_X="${4:-120}"
DATASET="volumes/raw"
MP_WORKERS=4
SCRIPT_PATH="$(dirname "$0")/clahe_gconn.py"

# Find all .zarr directories (zarrs are directories, not files)
mapfile -t ZARR_FILES < <(find "$FOLDER" -maxdepth 1 -name "*.zarr" -type d)

if [ ${#ZARR_FILES[@]} -eq 0 ]; then
    echo "No .zarr files found in: $FOLDER"
    exit 1
fi

echo "Found ${#ZARR_FILES[@]} zarr(s) to process."
echo "Kernel size: $KERNEL_Z $KERNEL_Y $KERNEL_X | Dataset: $DATASET | Workers: $MP_WORKERS"
echo "---"

FAILED=()

for ZARR in "${ZARR_FILES[@]}"; do
    # Strip trailing slash if present
    ZARR="${ZARR%/}"
    BASENAME="$(basename "$ZARR" .zarr)"
    OUTFILE="$(dirname "$ZARR")/${BASENAME}_clahe.zarr"

    echo "[$(date '+%H:%M:%S')] Processing: $ZARR"
    echo "  -> Output:  $OUTFILE"

    python "$SCRIPT_PATH" \
        -f "$ZARR" \
        -of "$OUTFILE" \
        -ds "$DATASET" \
        -k "$KERNEL_Z" "$KERNEL_Y" "$KERNEL_X" \
        -mp "$MP_WORKERS"

    if [ $? -ne 0 ]; then
        echo "  [ERROR] Failed on: $ZARR"
        FAILED+=("$ZARR")
    else
        echo "  [OK] Done: $OUTFILE"
    fi
    echo "---"
done

echo ""
echo "All done. ${#FAILED[@]} failure(s)."
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed files:"
    for F in "${FAILED[@]}"; do echo "  $F"; done
    exit 1
fi
