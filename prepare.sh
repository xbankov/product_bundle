#!/bin/bash

# Evaluate and find best params
python ./src/bundle_evaluation/evaluate_bundles_association_rules.py
python ./src/bundle_evaluation/evaluate_bundles_collaborative_filtering.py
python ./src/bundle_evaluation/evaluate_bundles_content_based_filtering.py
python ./src/evaluate_pricing.py

# Prepare static bundles
python ./src/generate_bundles/generate_bundles_association_rules.py
python ./src/generate_bundles/generate_bundles_collaborative_filtering.py
python ./src/generate_bundles/generate_bundles_content_based_filtering.py


python ./src/pricing/generate_pricing.py
