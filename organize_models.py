import os
import shutil
from pathlib import Path

# 1. Define your files and folder
files_to_move = [
    "recommendation_mappings.pkl",
    "drug_recommendation_model.keras"
]
target_folder = "drug_models_api"

# 2. Create the folder (if it doesn't exist)
Path(target_folder).mkdir(exist_ok=True)
print(f"📁 Created folder: {target_folder}")

# 3. Move each file
for filename in files_to_move:
    if os.path.exists(filename):
        shutil.move(filename, os.path.join(target_folder, filename))
        print(f"✅ Moved: {filename}")
    else:
        print(f"⚠️  Not found: {filename}")

# 4. Show final structure
print(f"\n🎯 Final structure in '{target_folder}':")
for item in os.listdir(target_folder):
    print(f"   - {item}")