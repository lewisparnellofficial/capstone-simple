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
    print()


def calculate_md5(file_path: Path) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def main() -> None:
    """Download and verify the CIC-IDS-2017 dataset."""

    zip_url = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip"
    md5_url = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.md5"

    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    zip_path = data_dir / "MachineLearningCSV.zip"
    md5_path = data_dir / "MachineLearningCSV.md5"

    print(f"Downloading {md5_url}...")
    urlretrieve(md5_url, md5_path)

    with open(md5_path, "r") as f:
        expected_hash = f.read().split()[0]

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        if zip_path.exists():
            print(f"Zip file already exists at {zip_path}")
            print("Verifying file integrity...")
            zip_hash = calculate_md5(zip_path)

            if zip_hash.lower() == expected_hash.lower():
                print("File integrity verified successfully!")
                break
            else:
                print(
                    f"ERROR: The zip file is corrupt (attempt {attempt}/{max_retries})"
                )
                print("Deleting corrupted file and redownloading...")
                zip_path.unlink()
        else:
            print(f"Download attempt {attempt}/{max_retries}")

        download_with_progress(zip_url, zip_path)

        print("Verifying file integrity...")
        zip_hash = calculate_md5(zip_path)

        if zip_hash.lower() == expected_hash.lower():
            print("File integrity verified successfully!")
            break
        else:
            if attempt < max_retries:
                print(
                    f"ERROR: Downloaded file is corrupt (attempt {attempt}/{max_retries})"
                )
                print("Deleting corrupted file and retrying...")
                zip_path.unlink()
            else:
                print(
                    f"ERROR: Failed to download valid file after {max_retries} attempts"
                )
                sys.exit(1)

    print(f"Extracting archive to {data_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    print()
    print("=" * 60)
    print("Download and extraction complete!")
    print()
    print("Next step: Run 'uv run ids-preprocess' to prepare the dataset")
    print("=" * 60)


if __name__ == "__main__":
    main()
