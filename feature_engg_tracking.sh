#!/usr/bin/env bash

# 1. Re-run feature engineering (uncomment when needed)
# python3 -m data_pipeline.feature_engg

# 2. DVC automatically detects changes
dvc status

# 3. Update DVC tracking
dvc add data/training_data_features.csv

# 4. Commit the new version
git add data/training_data_features.csv.dvc
git commit -m "feat: add time bucket open rates"
git tag -a "features-v1.3" -m "Added time bucket open rates"

# 5. Push
dvc push
git push --tags
