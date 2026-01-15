#!/bin/bash

cd /home/yuhwa2323/yuhwa/Lab/EDF/
export QLMPROJDIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "EDF Benchmarking"
python benchmarks/basic_test.py
bash benchmarks/kill_all.sh
sleep 5

cd ../FCFS/
echo "FCFS Benchmarking"
python benchmarks/basic_test.py
bash benchmarks/kill_all.sh
sleep 5

cd ../QLM_local
echo "QLM Benchmarking"
python benchmarks/basic_test.py
bash benchmarks/kill_all.sh
sleep 5

echo "Benchmark Finished"
