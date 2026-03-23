"""Download all required model weights for ScottNLP."""

import subprocess
import sys

def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, stdout=None, stderr=None)
    if result.returncode != 0:
        print(f"Command failed with exit code: {result.returncode}")
        sys.exit(1)

# 1. Install dependencies
run("pip install spacy sentence-transformers")

# 2. Download spaCy transformer model (~460 MB)
run("python -m spacy download en_core_web_trf")

# 3. Download Legal-BERT embedding model (~440 MB)
from sentence_transformers import SentenceTransformer
print("\n>>> Downloading nlpaueb/legal-bert-base-uncased ...")
model = SentenceTransformer('nlpaueb/legal-bert-base-uncased')
print("Legal-BERT download complete.")

# 4. Download Llama-3.1-8B-Instruct (~15 GB, used for Phase 1 section detection)
#    Optional: if you don't need Llama-assisted chunking, skip this step.
#    Phase 1 will automatically fall back to rule-based chunking.
#    After downloading, set LLAMA_MODEL_PATH in the .env file.
# from huggingface_hub import snapshot_download
# print("\n>>> Downloading Meta-Llama-3.1-8B-Instruct ...")
# snapshot_download(
#     repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
#     local_dir="model_weights/Meta-Llama-3.1-8B-Instruct",
# )
# print("Llama-3.1-8B-Instruct download complete.")
