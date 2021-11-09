#!/bin/bash

# Instructions taken from:
#   https://github.com/NVIDIA/cutlass/issues/154#issuecomment-745426099

# Set to persistent mode
nvidia-smi -i 0 -pm 1

# Lock the frequency to 900 MHz
sudo nvidia-smi -lgc 900 -i 0

# Monitor gpu clock, temperature, etc in 1000ms intervals. This needs to run in background during profiling to monitor the frequency is still locked.
screen -d -S mon -m bash -c "nvidia-smi --format=csv --query-gpu=clocks.sm,temperature.gpu,fan.speed,power.draw,power.limit -lms 1000 -i 0"
