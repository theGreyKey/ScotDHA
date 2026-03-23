#!/usr/bin/env python3
"""Convert all PDFs in ScotLaw/ to TXT files using PaddleOCR (GPU), saved in data/."""

import sys
from pathlib import Path
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import numpy as np

# Directories
SCOTLAW_DIR = Path(__file__).parent / "ScotLaw"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Initialize PaddleOCR with GPU
ocr = PaddleOCR(use_textline_orientation=True, lang="en", device="gpu", show_log=False)


def pdf_to_txt(pdf_path: Path, txt_path: Path):
    print(f"Processing: {pdf_path.name}")
    images = convert_from_path(str(pdf_path), dpi=200)
    all_lines = []

    for page_num, img in enumerate(images, start=1):
        print(f"  Page {page_num}/{len(images)}", end="\r", flush=True)
        img_array = np.array(img)
        result = ocr.predict(img_array)
        if result:
            for res in result:
                for text in res["rec_texts"]:
                    all_lines.append(text)
        all_lines.append("")  # Blank line between pages

    txt_path.write_text("\n".join(all_lines), encoding="utf-8")
    print(f"\n  -> Saved: {txt_path.name}")


def main():
    pdf_files = sorted(SCOTLAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in ScotLaw/")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF file(s). Output dir: {DATA_DIR}\n")

    for pdf_path in pdf_files:
        txt_name = pdf_path.stem + ".txt"
        txt_path = DATA_DIR / txt_name
        if txt_path.exists():
            print(f"Skipping (already exists): {txt_name}")
            continue
        try:
            pdf_to_txt(pdf_path, txt_path)
        except Exception as e:
            print(f"  ERROR processing {pdf_path.name}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
