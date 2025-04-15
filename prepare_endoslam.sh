#!/bin/bash

set -euo pipefail

ENDOSLAM_ROOT="EndoSlam"
OUTPUT_ROOT="prepared_endoslam"
LOG_FILE="processing.log"

echo "==== Processing Start: $(date) ====" > "$LOG_FILE"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

process_sequence() {
    local camera_type=$1
    local organ=$2
    local trajectory=$3
    local stl_file=$4

    log_message "Processing: $camera_type - $organ - $trajectory"

    # UnityCam-specific paths
    if [ "$camera_type" == "UnityCam" ]; then
        frames_dir="${ENDOSLAM_ROOT}/Cameras/${camera_type}/${organ}/${trajectory}/Frames"
        poses_dir="${ENDOSLAM_ROOT}/Cameras/${camera_type}/${organ}/${trajectory}/Poses"
        depth_dir="${ENDOSLAM_ROOT}/Cameras/${camera_type}/${organ}/${trajectory}/Pixelwise Depths"
        poses_file=$(find "$poses_dir" -maxdepth 1 -name '*.csv' -print -quit)
    # MiroCam-specific paths
    elif [ "$camera_type" == "MiroCam" ]; then
        frames_dir="${ENDOSLAM_ROOT}/Cameras/${camera_type}/${organ}/${trajectory}/Frames"
        poses_dir="${ENDOSLAM_ROOT}/Cameras/${camera_type}/${organ}/${trajectory}/Poses"
        depth_dir=""
        poses_file=$(find "$poses_dir" -maxdepth 1 -name '*.txt' -print -quit)
    else
        frames_dir="${ENDOSLAM_ROOT}/Cameras/${camera_type}/${organ}/${trajectory}/Frames"
        poses_dir="${ENDOSLAM_ROOT}/Cameras/${camera_type}/${organ}/${trajectory}/Poses"
        depth_dir=""
        poses_file=$(find "$poses_dir" -maxdepth 1 -name '*.xlsx' -print -quit)
    fi

    intrinsics_file="${ENDOSLAM_ROOT}/Cameras/${camera_type}/Calibration/cam.txt"
    output_dir="${OUTPUT_ROOT}/${camera_type}/${organ}/${trajectory}"

    # Validation checks
    [ -d "$frames_dir" ] || { log_message "ERROR: Missing frames directory"; return 1; }
    [ -f "$poses_file" ] || { log_message "ERROR: Poses file not found"; return 1; }
    [ -f "$intrinsics_file" ] || { log_message "ERROR: Intrinsics file missing"; return 1; }

    # Create output directory
    mkdir -p "$output_dir"/{images,poses,intrinsics,depths}

    # Run processor
    python endoslam_preprocess.py \
        --image_dir "$frames_dir" \
        --pose_file "$poses_file" \
        --intrinsics "$intrinsics_file" \
        --output_dir "$output_dir" \
        ${depth_dir:+--depth_dir "$depth_dir"} \
        ${stl_file:+--stl "$stl_file"}
}

# Main processing loop
find "${ENDOSLAM_ROOT}/Cameras" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' camera_type; do
    camera_name=$(basename "$camera_type")
    
    find "$camera_type" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' organ; do
        organ_name=$(basename "$organ")
        [ "$organ_name" == "Calibration" ] && continue
        
        stl_file="${ENDOSLAM_ROOT}/3D Scanners/${organ_name}/${organ_name}.stl"
        
        find "$organ" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' trajectory; do
            process_sequence "$camera_name" "$organ_name" "$(basename "$trajectory")" "$stl_file"
        done
    done
done

log_message "==== Processing Complete ===="
