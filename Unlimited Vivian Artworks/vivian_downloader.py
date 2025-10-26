

import os
import json
import time
from pixivpy3 import AppPixivAPI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------
# CONFIG
# ------------------------------------
SAVE_DIR = "./vivian_images"
TAG_JSON = "vivian_tags.json"
QUERY_TERMS = [
    "ビビアン",
    "Vivian",
    "ビビアン(ゼンゼロ)",
    "Vivian (Zenless Zone Zero)",
    "ビビアン・バンシー",
    "Vivian Banshee",
    "vivian",
    "薇薇安",
]
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_THREADS = 6
REFRESH_CHECK_INTERVAL = 20 * 60  # 20 minutes

# ------------------------------------
# LOGIN USING REFRESH TOKEN
# ------------------------------------
REFRESH_TOKEN = "5gYF_6tzWXRJnfgYTlnoPEUiqFRAoEobLWxVVGtnSeg"
api = AppPixivAPI()
api.auth(refresh_token=REFRESH_TOKEN)

# ------------------------------------
# LOAD EXISTING DATA (if any)
# ------------------------------------
if os.path.exists(TAG_JSON):
    with open(TAG_JSON, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
else:
    saved_data = []

all_data = {item["id"]: item for item in saved_data}
seen_ids = set(all_data.keys())

# ------------------------------------
# HELPER FUNCTION TO DOWNLOAD IMAGE
# ------------------------------------
def download_image(illust):
    img_id = illust.id
    title = illust.title
    tags = [t.name for t in illust.tags]
    filenames = []

    if illust.page_count == 1:
        filename = f"{SAVE_DIR}/{img_id}.jpg"
        if not os.path.exists(filename):
            api.download(illust.image_urls.large, path=SAVE_DIR, name=f"{img_id}.jpg")
        filenames.append(filename)
    else:
        for i, page in enumerate(illust.meta_pages):
            filename = f"{SAVE_DIR}/{img_id}_p{i+1}.jpg"
            if not os.path.exists(filename):
                api.download(page.image_urls.large, path=SAVE_DIR, name=f"{img_id}_p{i+1}.jpg")
            filenames.append(filename)
    return img_id, title, tags

# ------------------------------------
# DOWNLOAD BATCH WITH RETRY
# ------------------------------------
def download_batch(illusts, pbar):
    remaining = illusts.copy()
    while remaining:
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            future_to_illust = {executor.submit(download_image, illust): illust for illust in remaining}
            remaining = []
            for future in as_completed(future_to_illust):
                illust = future_to_illust[future]
                try:
                    img_id, title, tags = future.result()
                    if img_id in seen_ids:
                        all_data[img_id]["tags"] = sorted(list(set(all_data[img_id]["tags"] + tags)))
                    else:
                        all_data[img_id] = {
                            "id": img_id,
                            "title": title,
                            "tags": tags,
                            "url": f"https://www.pixiv.net/en/artworks/{img_id}"
                        }
                        seen_ids.add(img_id)
                    pbar.update(1)
                    pbar.set_postfix_str(title)
                except Exception as e:
                    print(f"⚠️ Failed {illust.id}: {e}. Will retry in 10s...")
                    remaining.append(illust)
        if remaining:
            time.sleep(10)  # wait before retrying failed images

# ------------------------------------
# MAIN DOWNLOAD LOOP
# ------------------------------------
last_refresh_check = time.time()

try:
    for term in QUERY_TERMS:
        print(f"\n🔍 Searching for: {term}")
        offset = 0
        pbar = tqdm(desc=f"Images for '{term}'", unit="img")
        while True:
            # Periodic refresh token check
            if time.time() - last_refresh_check > REFRESH_CHECK_INTERVAL:
                try:
                    api.search_illust("テスト", search_target='partial_match_for_tags', offset=0)
                    print("✅ Refresh token still valid.")
                except Exception:
                    print("⚠️ Refresh token expired or API login failed.")
                    REFRESH_TOKEN = input("Please enter new refresh token: ").strip()
                    api.auth(refresh_token=REFRESH_TOKEN)
                last_refresh_check = time.time()

            try:
                json_result = api.search_illust(term, search_target='partial_match_for_tags', offset=offset)
                if not json_result.illusts:
                    break
                # Filter out already seen images
                new_illusts = [illust for illust in json_result.illusts if illust.id not in seen_ids]
                if new_illusts:
                    download_batch(new_illusts, pbar)
                offset += len(json_result.illusts)
            except Exception as e:
                print(f"⚠️ Error during search or download: {e}. Retrying in 10s...")
                time.sleep(10)
        pbar.close()

except KeyboardInterrupt:
    print("\n⏹️ Download interrupted by user.")

# Save tag info
with open(TAG_JSON, "w", encoding="utf-8") as f:
    json.dump(list(all_data.values()), f, ensure_ascii=False, indent=2)

print(f"\n✅ All done. {len(all_data)} unique images saved in '{SAVE_DIR}', tag data in '{TAG_JSON}'.")
