#!/bin/bash
cd "."
if [[ "/media/fukuro/raid5/alltalk_tts" =~ " " ]]; then echo This script relies on Miniconda which can not be silently installed under a path with spaces. && exit; fi
# deactivate existing conda envs as needed to avoid conflicts
{ conda deactivate && conda deactivate && conda deactivate; } 2> /dev/null
# config
CONDA_ROOT_PREFIX="/media/fukuro/raid5/alltalk_tts/alltalk_environment/conda"
INSTALL_ENV_DIR="/media/fukuro/raid5/alltalk_tts/alltalk_environment/env"
# environment isolation
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
export CUDA_PATH="/media/fukuro/raid5/alltalk_tts/alltalk_environment/env"
export CUDA_HOME=""
# activate env
bash --init-file <(echo "source \"/media/fukuro/raid5/alltalk_tts/alltalk_environment/conda/etc/profile.d/conda.sh\" && conda activate \"/media/fukuro/raid5/alltalk_tts/alltalk_environment/env\"")
