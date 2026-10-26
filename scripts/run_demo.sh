#!/usr/bin/env bash
set -euo pipefail

python scripts/run_multi_person_sleep.py \
  --video data/demo/two_people_synthetic.mp4 \
  --names "Person A,Person B"

python scripts/render_multi_person_video.py \
  --video data/demo/two_people_synthetic.mp4
