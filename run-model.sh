#!/bin/bash
# 4 4 4 2 2 2 4 0 0
neighbour_window_sizes=(0 0 0)

for i in {0..9..3}
    do
        x=${neighbour_window_sizes[$i]}
        y=${neighbour_window_sizes[$i+1]}
        z=${neighbour_window_sizes[$i+2]}
        echo "Running model for x=$x, y=$y and z=$z and COS-INST-PHASE attribute..."
        
        if [[ -z $1 ]]; then
            echo "Scheduler's address not found. Running local version."
            docker run -it -v $(pwd):$(pwd) -e HOME=$(pwd) -w $(pwd) -u $(id -u):$(id -g) --network=host dasf-seismic:cpu python3 run-model.py --ml-model CIP-ml-model-$x-$y-$z.json --data data/F3_train.zarr --inline-window $x --trace-window $y --samples-window $z --output cos-inst-phase-$x-$y-$z.zarr
            echo "Local version ended."
        else
            echo "Running multi-node version for scheduler on $1..."
            docker run -it -v $(pwd):$(pwd) -e HOME=$(pwd) -w $(pwd) -u $(id -u):$(id -g) --network=host dasf-seismic:cpu python3 run-model.py --ml-model CIP-ml-model-$x-$y-$z.json --data data/F3_train.zarr --inline-window $x --trace-window $y --samples-window $z --address $1 --output cos-inst-phase-$x-$y-$z.zarr
            echo "Multi-node version ended."
        fi
        echo "Model trained."
        sleep 15
    done