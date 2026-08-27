from zipfile import ZipFile
import os
import sys
import time
import socket
import zipfile
import urllib.request

URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip' 
FILE = 'fashion_mnist_images.zip' 
FOLDER = 'fashion_mnist_images' 

def download_progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        shown = min(downloaded, total_size)
        sys.stdout.write(
            f'\rDownloading... {shown / (1024 * 1024):.1f} / {total_size / (1024 * 1024):.1f} MB'
            f' ({100 * shown // total_size}%)'
        )
    else:
        sys.stdout.write(f'\rDownloading... {downloaded / (1024 * 1024):.1f} MB')
    sys.stdout.flush()


def extract_with_progress(archive_path, dest):
    print(f'Extracting Fashion-MNIST dataset to {dest}...', flush=True)
    start_time = time.time()
    try:
        with ZipFile(archive_path) as zip_images:
            members = zip_images.infolist()
            total_files = len(members)
            print(f'Archive contains {total_files} files.', flush=True)
            for count, member in enumerate(members, 1):
                zip_images.extract(member, dest)
                if count % 2000 == 0 or count == total_files:
                    elapsed = time.time() - start_time
                    rate = count / elapsed if elapsed > 0 else 0.0
                    remaining = (total_files - count) / rate if rate > 0 else 0.0
                    percent = 100 * count // total_files if total_files else 100
                    print(
                        f'Extracted {count}/{total_files} files ({percent}%) - '
                        f'{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining',
                        flush=True,
                    )
    except zipfile.BadZipFile:
        os.remove(archive_path)
        raise SystemExit(
            'The dataset archive is corrupted (probably an interrupted download).\n'
            'It has been deleted - run the script again to re-download it.'
        )
    print(f'Extraction finished in {time.time() - start_time:.1f}s.', flush=True)


if not os.path.isfile(FILE):
    print(f'Downloading Fashion-MNIST dataset from {URL}...', flush=True)
    socket.setdefaulttimeout(60)  # fail instead of hanging forever on a stalled connection
    try:
        urllib.request.urlretrieve(URL, FILE, reporthook=download_progress_hook)
    except OSError as exc:
        if os.path.isfile(FILE):
            os.remove(FILE)
        raise SystemExit(
            f'Download failed: {exc}\n'
            'The partial file was deleted - run the script again to retry.'
        )
    sys.stdout.write('\n')
    sys.stdout.flush()
    print(f'Download finished ({os.path.getsize(FILE) / (1024 * 1024):.1f} MB).', flush=True)
else:
    print(f'Using existing archive {FILE} ({os.path.getsize(FILE) / (1024 * 1024):.1f} MB).', flush=True)
    
extract_with_progress(FILE, FOLDER)
    
print("DONE")