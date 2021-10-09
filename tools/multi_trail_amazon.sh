#!/bin/bash

PYTHON=${PYTHON:-"python"}

# $PYTHON tools/train.py configs/selfsup/pix_byol/pbyol_r18_sn6_sar_pro_ul_ep200_lr03.py

# $PYTHON tools/train.py configs/selfsup/pix_byol/pbyol_r18_sn6_sar_pro_ul_ep200_lr00375.py

$PYTHON tools/train.py configs/selfsup/pix_byol/pbyol_r18_sn6_sar_pro_ul_ep400_lr03.py