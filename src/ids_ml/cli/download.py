"""Download and verify the CIC-IDS-2017 dataset."""

import hashlib
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


def download_with_progress(url: str, destination: Path) -> None:
    """Download a file with a simple progress indicator."""
    print(f"Downloading {url}...")

    def report_progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            print(f"\rProgress: {percent:.1f}%", end="", flush=True)

    urlretrieve(url, destination, reporthook=report_progress)
    print()  # New line after progress


def calculate_md5(file_path: Path) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def main() -> None:
    """Download and verify the CIC-IDS-2017 dataset."""
    # URLs
    zip_url = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip"
    md5_url = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.md5"

    # Paths
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    zip_path = data_dir / "MachineLearningCSV.zip"
    md5_path = data_dir / "MachineLearningCSV.md5"

    # Download zip file if it doesn't exist
    if not zip_path.exists():
        download_with_progress(zip_url, zip_path)
    else:
        print(f"Zip file already exists at {zip_path}")

    # Always download the MD5 file to verify
    print(f"Downloading {md5_url}...")
    urlretrieve(md5_url, md5_path)

    # Verify MD5 hash
    print("Verifying file integrity...")
    zip_hash = calculate_md5(zip_path)

    with open(md5_path, "r") as f:
        expected_hash = f.read().split()[0]

    if zip_hash.lower() != expected_hash.lower():
        print("ERROR: The zip file is corrupt. Redownload the file.")
        sys.exit(1)

    print("File integrity verified successfully!")

    # Extract the archive
    print(f"Extracting archive to {data_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    print("Download and extraction complete!")


if __name__ == "__main__":
    main()
