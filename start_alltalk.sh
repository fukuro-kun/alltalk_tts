#!/bin/bash
source "/media/fukuro/raid5/alltalk_tts/alltalk_environment/conda/etc/profile.d/conda.sh"
conda activate "/media/fukuro/raid5/alltalk_tts/alltalk_environment/env"
export CUTLASS_PATH=/media/fukuro/raid5/cutlass
python script.py
