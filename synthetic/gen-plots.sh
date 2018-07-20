set -e
mkdir -p data/sparse
mkdir -p data/dense
mkdir -p data/trace
python ks-sparse.py
python ks-dense.py
python traceplots.py
mkdir -p plots
Rscript plotting.R
