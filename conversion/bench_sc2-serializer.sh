#!/bin/bash
start_time=$(date +%s)

for i in $(seq 0 15); do
    ./sc2_converter \
        --replays /home/bryce/SC2/replays/4.9.2/00004e179daa5a5bafeef22c01bc84408f70052a7e056df5c63800aed85099e9.SC2Replay \
        --output /home/bryce/SC2/converted/sc2_data_test_$i.SC2Replays --game /home/bryce/SC2/game/4.9.2/Versions \
        --converter action --port $((5670 + i)) >output_$i.txt 2>&1 &
done

wait

end_time=$(date +%s)
execution_time=$((end_time - start_time))

echo "Script execution time: $execution_time seconds"
