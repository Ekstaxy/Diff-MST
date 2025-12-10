import os
import re
import time
import shlex
import shutil
import zipfile
import argparse
import subprocess

from page_source import page_source
from utils import cprint, download_single_thread, download_smartdl, get_filename_from_url, remove_macos_hidden_files

def parse_args():
    parser = argparse.ArgumentParser(description="Download Cambridge-MT datasets")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the downloaded datasets")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers for downloading")
    parser.add_argument("--num_retry", type=int, default=5, help="Number of retries for downloading files")
    parser.add_argument("--start_idx", type=int, default=1, help="Start index for downloading files")

    return parser.parse_args()

def extract(output_path, url):
    """Extracts a zip file.

    Args:
        output_path (str): The directory to extract the files to.
        url (str): The URL of the zip file.

    Returns:
        bool: True if successful, False otherwise.
    """
    dl_filepath = os.path.join(output_path, get_filename_from_url(url))

    try:
        # Extract the downloaded zip file
        with zipfile.ZipFile(dl_filepath, 'r') as zip_file:
            # Create a subdirectory to extract files to
            zip_filename = os.path.splitext(os.path.basename(dl_filepath))[0]
            target_dir = os.path.join(output_path, zip_filename)
            subprocess.run(shlex.split(f'mkdir -p "{target_dir}"'))

            # Extract all files to target directory
            for file in zip_file.namelist():
                if not file.endswith('/'):  # skip directories
                    filename = os.path.basename(file)
                    with zip_file.open(file) as zipped_file, open(
                            os.path.join(target_dir, filename),
                            'wb') as extracted_file:
                        shutil.copyfileobj(zipped_file, extracted_file)

        # Remove macOS hidden files
        remove_macos_hidden_files(target_dir)

        # Delete the zip file after extraction
        os.remove(dl_filepath)

        return True
    
    except zipfile.BadZipFile:
        print(f'Failed to extract {dl_filepath}...')
        return False

def main():
    args = parse_args()

    urls = []
    cprint('Collecting urls from the page_source...', 'yellow')
    for d in page_source:
        if ('pt' in d.keys()) & ('dl' in d.keys()):
            if d['pt'].upper() == 'FULL':
                url = re.search('href="([^"]+)"', d['dl']).group(1)
                urls.append(url)

    n_items = len(urls)
    cprint(f'Found {n_items} full urls from the page_source...\n', 'yellow')

    n_successful_download = 0
    failed_urls = []
    failed_zips = []

    # Download files from the URL list
    for i, url in enumerate(urls[args.start_idx - 1:], start=args.start_idx - 1):
        cprint(f'[{i+1}/{n_items}] Downloading {url}...', 'green')
        if int(args.num_workers) == 1:
            result = download_single_thread(url, dl_dir=args.output_dir, n_retry=args.num_retry)
        else:
            result = download_smartdl(url, max_workers=int(args.num_workers), dl_dir=args.output_dir, n_retry=args.num_retry)

        if result is None:  # download was succesful
            time.sleep(1)
            if extract(args.output_dir, url):
                n_successful_download += 1
            else:
                failed_zips.append(url)
                cprint(f'Failed to extract {url}...')
        else:
            failed_urls.append(url)
            cprint(f'Failed to download {url} for {args.num_retry} tries... pass...')

    cprint(f'Finished downloading {n_successful_download} files.', 'white')
    cprint(f'Failed to download {len(failed_urls)} files.', 'white')
    cprint(f'Failed to extract {len(failed_zips)} files.', 'white')
    cprint(f'Failed urls: {failed_urls}', 'white')
    cprint(f'Failed zips: {failed_zips}', 'white')

if __name__ == "__main__":
    main()