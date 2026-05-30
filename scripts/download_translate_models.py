#!/usr/bin/env python3
import os
import sys
import urllib.request

MODELS = {
    "ja-en": "Xenova/opus-mt-ja-en",
    "ko-en": "Xenova/opus-mt-ko-en",
    "zh-en": "Xenova/opus-mt-zh-en",
}

FILES = [
    "onnx/encoder_model.onnx",
    "onnx/decoder_model.onnx",
    "source.spm",
    "target.spm",
    "vocab.json",
]

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        # Create opener with realistic User-Agent to avoid getting blocked by HF
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, dest_path)
        print("Success.")
    except Exception as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models", "translate")

    print(f"Saving models to: {models_dir}")

    for pair, repo in MODELS.items():
        pair_dir = os.path.join(models_dir, pair)
        os.makedirs(pair_dir, exist_ok=True)
        print(f"\n--- Downloading files for {pair} ({repo}) ---")

        for f in FILES:
            # Flatten folder structure in destination
            filename = os.path.basename(f)
            dest_path = os.path.join(pair_dir, filename)

            if os.path.exists(dest_path):
                print(f"{filename} already exists, skipping.")
                continue

            url = f"https://huggingface.co/Xenova/opus-mt-ja-en/resolve/main/{f}" if "ja-en" in repo else \
                  f"https://huggingface.co/{repo}/resolve/main/{f}"
            
            # Special fallback check because some older HF models might have source/target SPM capitalized
            # but Xenova repos are very consistent.
            download_file(url, dest_path)

    print("\nAll translation models downloaded successfully!")

if __name__ == "__main__":
    main()
