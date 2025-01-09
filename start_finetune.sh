#!/bin/bash
export TRAINER_TELEMETRY=0
source "/media/fukuro/raid5/alltalk_tts/alltalk_environment/conda/etc/profile.d/conda.sh"
conda activate "/media/fukuro/raid5/alltalk_tts/alltalk_environment/env"
export CUDA_VISIBLE_DEVICES=0  # Wählt die erste GPU
python finetune.py
