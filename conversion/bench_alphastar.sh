#!/bin/bash
# There's a slightly modified AlphaStar-Unplugged codebase in a private repo that contains extra logging
# code, adding some simple timing logs should be simple enough for users of this script to do...

start_time=$(date +%s)

export SC2PATH=/mnt/data/game/4.9.2

for i in $(seq 0 15); do
    mkdir -p /mnt/data/converted/tfrecord_$i
    ./data/generate_dataset.py \
        --sc2_replay_path /mnt/data/replays/4.9.2 \
        --converted_path /mnt/data/converted/tfrecord_$i \
        --partition_file /mnt/data/parts/bench_replay \
        --converter_settings configs/alphastar_supervised_converter_settings.pbtxt \
        --num_threads 1 >output_$i.txt 2>&1 &
done

wait

end_time=$(date +%s)
execution_time=$((end_time - start_time))

echo "Script execution time: $execution_time seconds"
