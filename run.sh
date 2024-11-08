#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate vlm_live_demo

python vlm_demo.py -i 0
