#!/bin/bash
set -e

rm -rf /tmp/drift-sense-check
rsync -a --exclude='.venv' --exclude='venv2' /mnt/c/Users/YUKANTH/drift_sense/ /tmp/drift-sense-check/
cd /tmp/drift-sense-check

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

echo "Prefix is: $(python3 -c 'import sys; print(sys.prefix)')"

pip install --upgrade pip
pip install -r requirements.txt
pip install pytest

echo "Running inference:"
python3 inference.py evaluation_dataset_stress/eval/sample_000/reference.png evaluation_dataset_stress/eval/sample_000/search.png

echo "Running pytest:"
PYTHONPATH="src" python3 -m pytest -q
