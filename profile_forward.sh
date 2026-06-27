#!/bin/bash

ncu \
    --set full \
    --kernel-name "cvmm_kernel" \
    --import-source yes \
    -o profile_forward \
    -f python3 benchmark.py