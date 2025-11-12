#@title download model /content/omniasr-transcriptions/server/download_models.sh

# %%writefile /content/omniasr-transcriptions/server/download_models.py
#!/usr/bin/env python3
"""
download_models.py
Ensures the MMS model files are downloaded into MODELS_DIR.
"""

import os
import urllib.request
import urllib.error
from tqdm.auto import tqdm
import sys


def download_file(url: str, download_file_path: str, redownload: bool = False) -> bool:
    """Download a single file with urllib + tqdm progress bar."""
    base_path = os.path.dirname(download_file_path)
    os.makedirs(base_path, exist_ok=True)

    # Skip if file already exists
    if os.path.exists(download_file_path):
        if redownload:
            os.remove(download_file_path)
            tqdm.write(f"♻️ Redownloading: {os.path.basename(download_file_path)}")
        elif os.path.getsize(download_file_path) > 0:
            tqdm.write(f"✔️ Skipped (already exists): {os.path.basename(download_file_path)}")
            return True

    # Try fetching metadata
    try:
        request = urllib.request.urlopen(url)
        total = int(request.headers.get("Content-Length", 0))
    except urllib.error.URLError as e:
        print(f"❌ Error: Unable to open URL: {url}")
        print(f"Reason: {e.reason}")
        return False

    # Download with progress bar
    with tqdm(
        total=total,
        desc=os.path.basename(download_file_path),
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress:
        try:
            urllib.request.urlretrieve(
                url,
                download_file_path,
                reporthook=lambda count, block_size, total_size: progress.update(block_size),
            )
        except urllib.error.URLError as e:
            print(f"❌ Error: Failed to download {url}")
            print(f"Reason: {e.reason}")
            return False

    tqdm.write(f"⬇️ Downloaded: {os.path.basename(download_file_path)}")
    return True


def main():
    # Use MODELS_DIR from environment variable or default
    MODELS_DIR = os.environ.get("MODELS_DIR", "./models")
    print(f"📁 Checking and downloading MMS models to: {MODELS_DIR}")

    # Check write permission
    if not os.access(os.path.dirname(MODELS_DIR) or ".", os.W_OK):
        print(f"✗ No write permission to {MODELS_DIR}")
        sys.exit(1)

    # ✅ Define URLs and build full local paths here
    model_urls = {
        "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/dictionary.txt":
            os.path.join(MODELS_DIR, "ctc_alignment_mling_uroman_model_dict.txt"),
        "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt":
            os.path.join(MODELS_DIR, "ctc_alignment_mling_uroman_model.pt"),
    }

    for url, full_path in model_urls.items():
        success = download_file(url, full_path)
        if not success:
            print(f"✗ Failed to fetch: {os.path.basename(full_path)}")
            sys.exit(1)

    print("✅ All model files are ready!")

main()
# if __name__ == "__main__":
#     main()
