#!/bin/bash

OPTIONS=$(getopt -o m:p:h --long model:,port:,help -- "$@")
if [ $? -ne 0 ]; then
  echo "Invalid arguments"
  exit 1
fi

eval set -- "$OPTIONS"

#MODEL="unsloth/Llama-3.2-1B-Instruct"
MODEL="meta-llama/Llama-3.2-1B-Instruct"
PORT="8000"

while true; do
  case "$1" in
  -m | --model)
    MODEL="$2"
    shift 2
    ;;
  -p | --port)
    PORT="$2"
    shift 2
    ;;
  -h | --help)
    echo "Usage: $0 --model <model_name> --port <port_number>"
    exit 0
    ;;
  --)
    shift
    break
    ;;
  *)
    echo "Unexpected option: $1"
    exit 1
    ;;
  esac
done

echo "Model: $MODEL"
echo "Port: $PORT"
vllm serve $MODEL --port $PORT --dtype=half --max-model-len 32768 --max-num-seqs 256 --scheduling-policy priority --preemption-mode swap --swap-space 16 --max-num-batched-tokens 131072 --gpu-memory-utilization 0.9
#vllm serve $MODEL --port $PORT --max-model-len 32768 #--gpu-memory-utilization 0.8
