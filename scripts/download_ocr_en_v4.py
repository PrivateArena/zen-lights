#!/usr/bin/env python3
import os
import sys
import urllib.request

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        # Set User-Agent to avoid blocks
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
    models_dir = os.path.join(base_dir, "models_ocr")
    os.makedirs(models_dir, exist_ok=True)
    
    model_url = "https://huggingface.co/breezedeus/cnocr-ppocr-en_PP-OCRv4/resolve/main/en_PP-OCRv4_rec_infer.onnx"
    dict_url = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/en_dict.txt"
    
    model_dest = os.path.join(models_dir, "en_PP-OCRv4_rec_infer.onnx")
    dict_dest = os.path.join(models_dir, "en_dict.txt")
    
    print("--- Downloading English PP-OCRv4 Model ---")
    download_file(model_url, model_dest)
    
    print("\n--- Downloading English Dictionary ---")
    download_file(dict_url, dict_dest)
    
    print("\nAll English OCR assets downloaded successfully!")

if __name__ == "__main__":
    main()
