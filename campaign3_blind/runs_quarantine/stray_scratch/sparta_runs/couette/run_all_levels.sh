#!/bin/bash

cd /home/alexander/Schreibtisch/sparta_runs/couette

SPARTA=/home/alexander/Schreibtisch/sparta/src/spa_serial

for level in 1 2 3; do
    echo "Running Level $level..."
    rm -f surf_output.* log.sparta
    
    $SPARTA -i in.couette_level$level > log.sparta 2>&1
    
    # Copy results for this level
    if [ -f surf_output.30000 ]; then
        cp surf_output.30000 surf_output_level${level}.30000
        cp log.sparta log_level${level}.sparta
        echo "Level $level completed successfully"
    else
        echo "ERROR: Level $level failed - no output file"
    fi
done

echo "All levels completed"
