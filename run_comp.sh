#!/bin/bash
set -x
trap '' HUP   # survive launcher/ssh death; children inherit the ignored disposition
mkdir -p results real_data/cadence real_data/rfi
exec > >(tee results/train.log) 2>&1
python3 -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
sudo apt-get install -y build-essential python3-dev unzip 2>&1 | tail -1
# numpy 2.x on the Lambda image breaks blimpy/setigen -> pin numpy<2 FIRST, then the SETI stack
pip3 install -q --user "numpy<2.0"
pip3 install -q --user hdf5plugin h5py "astropy<6" scipy matplotlib
pip3 install -q --user "scipy>=1.11.2"
pip3 install -q --user blimpy setigen turbo_seti
python3 - <<'PY'
import numpy; print("numpy", numpy.__version__)
from blimpy import Waterfall; import setigen, hdf5plugin, torch
print("SETI STACK OK; cuda", torch.cuda.is_available())
PY
if [ $? -ne 0 ]; then echo "STACK_FAIL -> force reinstall"; pip3 install -q --user --force-reinstall "numpy<2.0" blimpy==2.1.4 setigen==2.7.0; python3 -c "from blimpy import Waterfall; print('retry ok')"; fi
echo "=== download REAL BL data ==="
CB=http://blpd14.ssl.berkeley.edu/voyager_2020/single_coarse_channel
for f in single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5 \
         single_coarse_guppi_59046_80354_DIAG_VOYAGER-1_0012.rawspec.0000.h5 \
         single_coarse_guppi_59046_80672_DIAG_VOYAGER-1_0013.rawspec.0000.h5 \
         single_coarse_guppi_59046_80989_DIAG_VOYAGER-1_0014.rawspec.0000.h5 \
         single_coarse_guppi_59046_81310_DIAG_VOYAGER-1_0015.rawspec.0000.h5 \
         single_coarse_guppi_59046_81628_DIAG_VOYAGER-1_0016.rawspec.0000.h5; do
  curl -s -o real_data/cadence/$f $CB/$f; done
EB=http://blpd0.ssl.berkeley.edu/lband2017/top11hdf5
for t in HIP82860 HIP66704 HIP7981 HIP65352; do curl -s -o real_data/rfi/$t.zip $EB/$t.zip; (cd real_data/rfi && unzip -oq $t.zip); done
echo "cadence=$(ls real_data/cadence/*.h5|wc -l) rfi=$(ls -d real_data/rfi/HIP*/|wc -l)"
python3 driftnet_comprehensive.py
touch /home/ubuntu/job/JOB_DONE   # fire the instance finalizer independent of the launcher
echo ALL_DONE
