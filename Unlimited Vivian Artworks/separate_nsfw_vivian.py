import os
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ------------------------------------
# CONFIG
# ------------------------------------
SOURCE_DIR = "./vivian_images"
NSFW_DIR = "./vivian_nsfw_tagged"
SAFE_DIR = "./vivian_safe_tagged"
JSON_FILE = "vivian_tags.json"
MAX_THREADS = 6

os.makedirs(NSFW_DIR, exist_ok=True)
os.makedirs(SAFE_DIR, exist_ok=True)

# ------------------------------------
# LOAD JSON DATA
# ------------------------------------
with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# ------------------------------------
# HELPER FUNCTION TO COPY FILES
# ------------------------------------
def copy_img_files(entry):
    """
    Copies all images associated with an img id to the target folder
    depending on whether "R-18" tag is present.
    """
    img_id = entry["id"]
    tags = entry["tags"]
    target_dir = NSFW_DIR if "R-18" in tags else SAFE_DIR

    copied_files = 0
    # Search for files starting with img_id
    for fname in os.listdir(SOURCE_DIR):
        if fname.startswith(str(img_id)):
            src_path = os.path.join(SOURCE_DIR, fname)
            dst_path = os.path.join(target_dir, fname)
            shutil.copy2(src_path, dst_path)
            copied_files += 1
    return copied_files

# ------------------------------------
# MULTITHREADING WITH PROGRESS BAR
# ------------------------------------
total_entries = len(data)
pbar = tqdm(total=total_entries, desc="Copying images", unit="img_id")

with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    futures = [executor.submit(copy_img_files, entry) for entry in data]
    for future in as_completed(futures):
        _ = future.result()  # just to catch exceptions if any
        pbar.update(1)

pbar.close()
print("\n✅ All images copied successfully!")
print(f"NSFW images in '{NSFW_DIR}', safe images in '{SAFE_DIR}'.")
