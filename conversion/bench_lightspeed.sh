#!/bin/bash
start_time=$(date +%s)

for i in $(seq 0 15); do
    ./build/Release/lightspeed -p $((5670 + i)) >output_$i.txt 2>&1 &
done

wait

end_time=$(date +%s)
execution_time=$((end_time - start_time))

echo "Script execution time: $execution_time seconds"
