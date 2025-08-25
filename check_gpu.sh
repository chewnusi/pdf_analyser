#!/bin/bash
# check_gpu.sh
nvidia_smi_path=$(find /usr /opt -name "nvidia-smi" -type f -executable 2>/dev/null | head -n1)

if [[ -n "$nvidia_smi_path" ]] && "$nvidia_smi_path" &> /dev/null; then
  echo "-f docker-compose.yml -f docker-compose.gpu.yml"
else
  echo "-f docker-compose.yml"
fi