#!/bin/bash

# Evaluate, find best params and prepare static files
python ./scripts/evaluate_bundles_association_rules.py
python ./scripts/evaluate_bundles_collaborative_filtering.py
python ./scripts/evaluate_bundles_content_based_filtering.py
python ./scripts/evaluate_bundles_all.py
python ./scripts/pricing_train_regression.py
python ./scripts/generate_pricing.py
