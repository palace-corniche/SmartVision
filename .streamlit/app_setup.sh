#!/bin/bash
# Force removal of opencv-python (which causes libGL errors)
pip uninstall -y opencv-python opencv-contrib-python || true
pip install opencv-python-headless==4.9.0.80
