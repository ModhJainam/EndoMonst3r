#!/bin/bash
# process_c3vd.sh - Process all C3VD trajectory folders

set -eo pipefail

# Configuration
SCRIPT_DIR=$(dirname "$(realpath "$0")")
INPUT_DIR="${SCRIPT_DIR}/c3vd"  # Default input directory
OUTPUT_ROOT="${SCRIPT_DIR}/processed_c3vd"
LOG_FILE="${SCRIPT_DIR}/c3vd_processing.log"
PARALLEL_JOBS=1  # Default sequential processing

# Initialize logging
exec > >(tee -a "$LOG_FILE") 2>&1
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Usage information
show_help() {
    echo "Usage: $0 [-i input_dir] [-o output_root] [-j parallel_jobs]"
    echo "Process C3VD trajectories using c3vd_preprocess.py"
    echo "Default locations:"
    echo "  Input: ${INPUT_DIR}"
    echo "  Output: ${OUTPUT_ROOT}"
    exit 0
}

# Parse arguments
while getopts "i:o:j:h" opt; do
    case "$opt" in
        i) INPUT_DIR=$(realpath "$OPTARG") ;;
        o) OUTPUT_ROOT=$(realpath "$OPTARG") ;;
        j) PARALLEL_JOBS="$OPTARG" ;;
        h) show_help ;;
        *) exit 1 ;;
    esac
done

# Validate input
if [[ ! -d "$INPUT_DIR" ]]; then
    log "Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Create output root
mkdir -p "$OUTPUT_ROOT"

# Processing function
process_trajectory() {
    local traj_path="$1"
    local traj_name=$(basename "$traj_path")
    local output_dir="${OUTPUT_ROOT}/${traj_name}"
    
    log "Processing: ${traj_name}"
    
    # Create output directory
    mkdir -p "$output_dir" || {
        log "Error: Failed to create output directory: ${output_dir}"
        return 1
    }
    
    # Run preprocessing
    python "${SCRIPT_DIR}/c3vd_preprocess.py" \
        --input_dir "$traj_path" \
        --output_dir "$output_dir" || {
        log "Error: Failed to process ${traj_name}"
        return 1
    }
    
    log "Completed: ${traj_name}"
}

# Main processing
export SCRIPT_DIR OUTPUT_ROOT LOG_FILE
export -f process_trajectory log

log "==== Starting C3VD Processing ===="
log "Input Directory: ${INPUT_DIR}"
log "Output Root: ${OUTPUT_ROOT}"

find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | \
    xargs -0 -I{} -P "$PARALLEL_JOBS" bash -c 'process_trajectory "$@"' _ {}

log "==== Processing Complete ===="
