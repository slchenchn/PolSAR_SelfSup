#!/bin/bash

PYTHON=${PYTHON:-"python"}

$PYTHON tools/train.py configs/selfsup/pix_byol/pbyol_r50_sn6_sar_pro_ep200.py