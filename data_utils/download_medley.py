import os
import shlex
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Download MedleyDB datasets")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the downloaded datasets")

    return parser.parse_args()

def main():
    args = parse_args()

    # MedleyDB 1.0
    subprocess.run(shlex.split(f"kaggle datasets download -d ajchen2005/medleydb-v1 -p {os.path.join(args.output_dir, 'MedleyDB_V1')}"))
    subprocess.run(shlex.split(f"unzip {os.path.join(args.output_dir, 'MedleyDB_V1/medleydb-v1.zip')}"))
    subprocess.run(shlex.split(f"rm {os.path.join(args.output_dir, 'MedleyDB_V1/medleydb-v1.zip')}"))

    # MedleyDB 2.0
    subprocess.run(shlex.split(f"kaggle datasets download -d ajchen2005/medley-db-v2 -p {os.path.join(args.output_dir, 'MedleyDB_V2')}"))
    subprocess.run(shlex.split(f"unzip {os.path.join(args.output_dir, 'MedleyDB_V2/medley-db-v2.zip')}"))
    subprocess.run(shlex.split(f"rm {os.path.join(args.output_dir, 'MedleyDB_V2/medley-db-v2.zip')}"))

if __name__ == "__main__":
    main()