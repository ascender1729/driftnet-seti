#!/bin/bash
set -x
trap '' HUP
mkdir -p results real_data/cadence real_data/rfi
exec > >(tee results/cp4.log) 2>&1
python3 -c "import torch; print('CUDA', torch.cuda.is_available())"
sudo apt-get install -y build-essential python3-dev unzip 2>&1 | tail -1
pip3 install -q --user "numpy<2.0"
pip3 install -q --user hdf5plugin h5py "astropy<6" "scipy>=1.11.2" matplotlib
pip3 install -q --user blimpy setigen turbo_seti
python3 -c "from blimpy import Waterfall; import setigen,turbo_seti,torch; print('STACK OK cuda',torch.cuda.is_available())" || { echo STACK_FAIL; pip3 install -q --user --force-reinstall "numpy<2.0" blimpy==2.1.4 setigen==2.7.0 turbo_seti; }
echo "=== download Voyager X-band cadence (6 files) ==="
CB=http://blpd14.ssl.berkeley.edu/voyager_2020/single_coarse_channel
for f in single_coarse_guppi_59046_80036_DIAG_VOYAGER-1_0011.rawspec.0000.h5 \
         single_coarse_guppi_59046_80354_DIAG_VOYAGER-1_0012.rawspec.0000.h5 \
         single_coarse_guppi_59046_80672_DIAG_VOYAGER-1_0013.rawspec.0000.h5 \
         single_coarse_guppi_59046_80989_DIAG_VOYAGER-1_0014.rawspec.0000.h5 \
         single_coarse_guppi_59046_81310_DIAG_VOYAGER-1_0015.rawspec.0000.h5 \
         single_coarse_guppi_59046_81628_DIAG_VOYAGER-1_0016.rawspec.0000.h5; do
  curl -s -o real_data/cadence/$f $CB/$f; done
EB=http://blpd0.ssl.berkeley.edu/lband2017/top11hdf5
curl -s -o real_data/rfi/HIP7981.zip $EB/HIP7981.zip; (cd real_data/rfi && unzip -oq HIP7981.zip)
echo "cadence=$(ls real_data/cadence/*.h5|wc -l) rfi=$(ls real_data/rfi/*/*.h5 2>/dev/null|wc -l)"
echo "=== CP4 QUICK VALIDATION (X band) ==="
python3 cp4_cadence_roc.py --cad-dir real_data/cadence --band X --quick
echo "=== CP4 QUICK (L band, Enriquez) ==="
python3 cp4_cadence_roc.py --cad-dir real_data/rfi --band L --quick
touch /home/ubuntu/job/JOB_DONE
echo ALL_DONE
