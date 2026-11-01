"""
Dataset acquisition module for the PlantVillage dataset.
Downloads genuine images for target disease and healthy classes directly from the official repository.
"""

import os
import json
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import io

TARGET_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
]

SAMPLES_PER_CLASS = 80
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "raw")
GITHUB_TREE_URL = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/git/trees/master?recursive=1"
RAW_BASE_URL = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/"


def fetch_file_list():
    print(f"Connecting to PlantVillage repository tree...")
    req = urllib.request.Request(GITHUB_TREE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    return data.get("tree", [])


def download_single_image(rel_path, target_path):
    encoded_path = urllib.parse.quote(rel_path)
    url = RAW_BASE_URL + encoded_path
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            content = resp.read()
        # Verify image integrity
        img = Image.open(io.BytesIO(content))
        img.verify()
        # Save verified image
        with open(target_path, "wb") as f:
            f.write(content)
        return True, target_path
    except Exception as e:
        return False, f"{rel_path}: {e}"


def download_dataset():
    os.makedirs(RAW_DIR, exist_ok=True)
    tree = fetch_file_list()

    download_tasks = []
    class_stats = {}

    for cls in TARGET_CLASSES:
        cls_dir = os.path.join(RAW_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
        # Filter matching paths
        pattern = f"raw/color/{cls}/"
        cls_files = [x["path"] for x in tree if pattern in x["path"] and (x["path"].endswith(".JPG") or x["path"].endswith(".jpg"))]
        cls_files = sorted(cls_files)[:SAMPLES_PER_CLASS]
        class_stats[cls] = len(cls_files)
        print(f"Identified {len(cls_files)} genuine image candidates for [{cls}]")

        for idx, file_path in enumerate(cls_files):
            fname = f"{cls}_{idx:03d}.jpg"
            dest = os.path.join(cls_dir, fname)
            if not os.path.exists(dest) or os.path.getsize(dest) == 0:
                download_tasks.append((file_path, dest))

    print(f"\nCommencing parallel download for {len(download_tasks)} files...")
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_single_image, path, dest): (path, dest) for path, dest in download_tasks}
        for future in as_completed(futures):
            ok, msg = future.result()
            if ok:
                success_count += 1
                if success_count % 20 == 0 or success_count == len(download_tasks):
                    print(f"Downloaded & verified {success_count}/{len(download_tasks)} images...")
            else:
                fail_count += 1
                print(f"Download warning: {msg}")

    print(f"\nDownload summary: {success_count} succeeded, {fail_count} failed.")
    print("Class distribution in raw storage:")
    for cls in TARGET_CLASSES:
        cls_dir = os.path.join(RAW_DIR, cls)
        count = len([f for f in os.listdir(cls_dir) if f.endswith(".jpg") or f.endswith(".JPG")])
        print(f" - {cls}: {count} images")


if __name__ == "__main__":
    download_dataset()
