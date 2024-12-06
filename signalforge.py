import os
import sys
import shutil
import time
import subprocess
from argparse import ArgumentParser

# Define colors
RED = '\033[0;31m'
BLUE = '\033[0;94m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
NC = '\033[0m'  # No Color

def translate_windows_to_linux_path(windows_path):
    """
    Converts a Windows file path to a Linux file path, handling whitespaces.
    
    Args:
        windows_path (str): The Windows file path to convert.
    
    Returns:
        str: The converted Linux file path.
    """
    windows_path = windows_path.strip()
    linux_path = windows_path.replace("\\", "/")
    if ":" in linux_path:
        drive, path = linux_path.split(":", 1)
        linux_path = f"/mnt/{drive.lower()}{path}"
    return linux_path

# Argument parsing
parser = ArgumentParser(description="Process input and output directories with specified mode.")
parser.add_argument("mode", choices=["hel", "pol"], help="Mode of operation ('hel' or 'pol').")
parser.add_argument("input_dirs", nargs='+', help="List of input directories (Windows or Linux paths).")
parser.add_argument("output_dir", help="Path to the output directory (Windows or Linux path).")
parser.add_argument("--skip-c", action="store_true", 
                    help="Skip the 'make clean all' and 'signal_to_text' steps and only run signal_to_psd.")

args = parser.parse_args()
mode = args.mode
input_dirs = [translate_windows_to_linux_path(d) for d in args.input_dirs]
output_dir = translate_windows_to_linux_path(args.output_dir)
print(f"input_dirs: {args.input_dirs} -> {input_dirs}")
print(f"output_dir: {args.output_dir} -> {output_dir}")

# Start timer
start_time = time.time()

# Clean
# if os.path.exists(output_dir):
#     response = input(f"{RED}Do you want to delete its contents? (yes/no): {NC} ").strip().lower()
#     if response in ['yes', 'y']:
#         print(f"{YELLOW}Removing old contents of the output directory...{NC}")
#         shutil.rmtree(output_dir)
#     else:
#         print(f"{YELLOW}{output_dir} directory contents remain intact.{NC}")

if not args.skip_c:
    print(f"{YELLOW}Running make clean all...{NC}")
    subprocess.run(["make", 
                    # "-s", 
                    "clean", "all", "-C", "./c"], check=True)

    # Run signal_to_text
    print(f"{YELLOW}Running signal_to_text...{NC}")
    signal_to_text_cmd = ["./c/build/signal_to_text", mode] + input_dirs + [output_dir]
    try:
        subprocess.run(signal_to_text_cmd, check=True)
        print(f"{GREEN}signal_to_text execution complete.{NC}")
    except subprocess.CalledProcessError as e:
        print(f"{RED}signal_to_text failed: {e}{NC}")
        sys.exit(1)

# Run signal_to_psd.py
print(f"{YELLOW}Running signal_to_psd...{NC}")
signal_to_psd_cmd = ["python3", "./py/signal_to_psd.py", "--file-type", mode, "-d", output_dir]
try:
    subprocess.run(signal_to_psd_cmd, check=True)
    print(f"{GREEN}signal_to_psd execution complete.{NC}")
except subprocess.CalledProcessError as e:
    print(f"{RED}signal_to_psd failed: {e.returncode}{NC}")
    sys.exit(e.returncode)

# Calculate execution time
end_time = time.time()
execution_time = int(end_time - start_time)
print(f"{BLUE}Execution completed in {execution_time} seconds.{NC}")
