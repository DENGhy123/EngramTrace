from pathlib import Path
from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "libero"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading LIBERO-Spatial...")
print("Target directory:", DOWNLOAD_DIR)

snapshot_download(
    repo_id="yifengzhu-hf/LIBERO-datasets",
    repo_type="dataset",
    allow_patterns=["libero_spatial/*"],
    local_dir=str(DOWNLOAD_DIR),
)

print("Download finished.")