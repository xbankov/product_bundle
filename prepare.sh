#!/bin/bash

# Evaluate, find best params and prepare static files
python ./src/evaluate_bundles_association_rules.py
python ./src/evaluate_bundles_collaborative_filtering.py
python ./src/evaluate_bundles_content_based_filtering.py
python ./src/evaluate_bundles_all.py
python ./src/evaluate_pricing.py
python ./src/generate_pricing.py
