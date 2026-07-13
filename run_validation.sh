#!/bin/bash
# Validation pipeline: hard negatives (X+L), reverse OOD (L->X), Voyager real-signal control.
# No set -e (one stage failure must not lose the rest). Always collect a log + whatever results exist.
set -x
trap '' HUP
cd "$(dirname "$0")"
mkdir -p results real_data/cadence real_data/rfi
exec > >(tee results/validation.log) 2>&1
python3 -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
sudo apt-get update -y 2>&1 | tail -1
sudo apt-get install -y build-essential python3-dev unzip 2>&1 | tail -1
pip3 install -q --user "numpy<2.0"
pip3 install -q --user hdf5plugin h5py "astropy<6" "scipy>=1.11.2" scikit-learn matplotlib
pip3 install -q --user blimpy setigen turbo_seti
python3 - <<'PY'
from blimpy import Waterfall
import setigen, hdf5plugin, torch, sklearn
print("SETI STACK OK; cuda", torch.cuda.is_available())
PY
if [ $? -ne 0 ]; then echo "STACK_FAIL -> force reinstall"; pip3 install -q --user --force-reinstall "numpy<2.0" blimpy==2.1.4 setigen==2.7.0 turbo_seti; python3 -c "from blimpy import Waterfall; print('retry ok')"; fi

echo "=== download REAL BL data (Voyager X + 4 Enriquez L cadences) ==="
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
echo "cadence=$(ls real_data/cadence/*.h5|wc -l) rfi_cadences=$(ls -d real_data/rfi/HIP*/ 2>/dev/null|wc -l)"

QUICK="${QUICK:-}"      # export QUICK=--quick for a smoke test
run(){ echo "===== $* ====="; python3 "$@" 2>&1 | tail -40 || echo "STAGE_FAILED: $*"; }

run hardneg.py --cad-dir real_data/cadence --band X --trials 24 $QUICK
run hardneg.py --cad-dir real_data/rfi     --band L --trials 24 $QUICK
run reverse_ood.py
run voyager_realsignal.py --cad-dir real_data/cadence

echo "=== collect ==="
cp -f results/hardneg/hardneg_results_*.json results/ 2>/dev/null
find results -maxdepth 2 -name "*.json" | sort
touch JOB_DONE
echo VALIDATION_ALL_DONE
