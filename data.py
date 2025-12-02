import os
import shutil

BASE = r"C:\Users\PTCL\.vscode\X_ray"   # CHANGE THIS TO YOUR PATH

# -------------------------------------------------------
# 1. Delete macOS junk files and folders
# -------------------------------------------------------
junk_files = [".DS_Store", "._DS_Store", "._train", "._test", "._val", "._chest_xray"]
junk_dirs = ["__MACOSX"]

for root, dirs, files in os.walk(BASE, topdown=False):
    for f in files:
        if f in junk_files or f.startswith("._"):
            try:
                os.remove(os.path.join(root, f))
            except:
                pass
    for d in dirs:
        if d in junk_dirs:
            try:
                shutil.rmtree(os.path.join(root, d))
            except:
                pass

print("✔ Removed macOS junk")

# -------------------------------------------------------
# 2. Find the REAL chest_xray folder
# -------------------------------------------------------
real_path = None
for root, dirs, _ in os.walk(BASE):
    if "train" in dirs and "test" in dirs and "val" in dirs:
        real_path = root
        break

if real_path is None:
    raise Exception("❌ Could not locate the correct chest_xray/train|test|val folder")

print("✔ Found dataset at:", real_path)

# -------------------------------------------------------
# 3. Create clean final folder
# -------------------------------------------------------
final_path = os.path.join(BASE, "chest_xray_clean")
os.makedirs(final_path, exist_ok=True)

for split in ["train", "test", "val"]:
    shutil.copytree(
        os.path.join(real_path, split),
        os.path.join(final_path, split),
        dirs_exist_ok=True
    )

print("✔ Copied dataset into clean folder")

print("\n🎉 CLEANUP COMPLETE!")
print("Your final dataset is here:", final_path)
