#!/bin/bash

python ./src/rulebased_hyperoptimization.py
python ./src/generate_bundles.py
python ./src/generate_pricing.py
python ./src/generate_similarity.py
