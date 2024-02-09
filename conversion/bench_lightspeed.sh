#!/bin/bash
start_time=$(date +%s)

for i in $(seq 0 15); do
    ./lightspeed \
        --replay /home/bryce/SC2/replays/4.9.2/00004e179daa5a5bafeef22c01bc84408f70052a7e056df5c63800aed85099e9.SC2Replay \
        --game /home/bryce/SC2/game/4.9.2/Versions/Base74741/SC2_x64 -p $((5670 + i)) >output_$i.txt 2>&1 &
done

wait

end_time=$(date +%s)
execution_time=$((end_time - start_time))

echo "Script execution time: $execution_time seconds"
