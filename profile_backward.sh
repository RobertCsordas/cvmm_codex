#!/bin/bash

ncu \
    --set full \
    --kernel-name "cvmm_backward_kernel3" \
    --import-source yes \
    -o profile_backward \
    -f python3 benchmark.py