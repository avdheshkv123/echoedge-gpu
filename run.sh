#!/bin/bash
KERNEL=$1
BLOCK=16
INPUT=data/input
OUTPUT=results

./echoedge_gpu.exe -k $KERNEL -b $BLOCK -i $INPUT -o $OUTPUT
echo "DONE. Output written to $OUTPUT/"
