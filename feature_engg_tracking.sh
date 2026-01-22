#/bin/bash

# 1. Modify feature_engg.py with new features
# 2. Re-run feature engineering
python feature_engg.py

# 3. DVC automatically detects changes
dvc status  # Shows modified files

# 4. Update DVC tracking
dvc add data/training_data_features.csv

# 5. Commit the new version
git add data/training_data_features.csv.dvc
git commit -m "feat: add interaction features + drop hour column"
git tag -a "features-v1.1" -m "Added interaction features + dropped hour column"

# 6. Push
dvc push
git push --tags