#!/bin/bash

exec 1>/output/stdout.log 2>/output/stderr.log

mkdir -p /output

MODELPATH=$1

# Run the load model python script
python3 inference_cpu_model.py  --model_path ${MODELPATH}

cd /output

tar zcvf output.tgz stdout.log stderr.log