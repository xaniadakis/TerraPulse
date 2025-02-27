import os
import sys
import shutil
import time
import subprocess
from argparse import ArgumentParser
from datetime import datetime

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


# Define colors
RED = '\033[0;31m'
BLUE = '\033[0;94m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
NC = '\033[0m'  # No Color

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8', errors='replace', closefd=False)
 
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


    # Define the 'runs' directory path
    runs_dir = os.path.join(os.path.dirname(output_dir), "run_logs")

    # Create the 'runs' directory if it doesn't exist
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
        print(f"'runs' directory created at: {runs_dir}")

    # Generate log file path based on datetime
    log_file_name = datetime.now().strftime("%Y%m%d_%H%M.log")
    c_log_file_path = os.path.join(runs_dir, "C_"+log_file_name)
    py_log_file_path = os.path.join(runs_dir, "py_"+log_file_name)
    print(f"Log files will be created at: {runs_dir}")

    # Clean
    # if os.path.exists(output_dir):
    #     response = input(f"{RED}Do you want to delete its contents? (yes/no): {NC} ").strip().lower()
    #     if response in ['yes', 'y']:
    #         print(f"{YELLOW}Removing old contents of the output directory...{NC}")
    #         shutil.rmtree(output_dir)
    #     else:
    #         print(f"{YELLOW}{output_dir} directory contents remain intact.{NC}")

    try:
        if not args.skip_c:
            print(f"{YELLOW}Running make clean all...{NC}")
            # subprocess.run(["make", 
            #                 # "-s", 
            #                 "clean", "all", "-C", "./c"], check=True)
            process = subprocess.Popen(["make", 
                            # "-s", 
                            "clean", "all", "-C", "./c"], stdout=sys.stdout, stderr=sys.stderr)
            process.wait()
            print(f"{YELLOW}Running signal_to_text...{NC}")
            # cmd = ["./c/build/signal_to_text", mode, c_log_file_path] + input_dirs + [output_dir]
            
            # with open(c_log_file_path, "w") as c_log_file:
            #     c_log_file.write(f"C Log started on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} with cmd: {cmd}\n")
            # with open(c_log_file_path, "a") as log_file:
            #     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            #     tee_process = subprocess.Popen(["tee", "-a", c_log_file_path], stdin=process.stdout)
            #     process.stdout.close()  # Allow tee to finish reading
            #     process.wait()
            #     tee_process.wait() 


            # Open the log file once at the start
            with open(c_log_file_path, "w") as c_log_file:
                c_log_file.write(f"C Log started on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # Loop over each input directory and run the C executable separately
            for input_dir in input_dirs:
                cmd = ["./c/build/signal_to_text", mode, c_log_file_path, input_dir, output_dir]

                # Append command execution details to the log file
                with open(c_log_file_path, "a") as c_log_file:
                    c_log_file.write(f"|RUN|: {input_dir} TO {output_dir}\n")

                # Execute and pipe through tee
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                tee_process = subprocess.Popen(["tee", "-a", c_log_file_path], stdin=process.stdout)
                
                process.stdout.close()  # Allow tee to finish reading
                process.wait()
                tee_process.wait()

                with open(c_log_file_path, "a") as c_log_file:
                    c_log_file.write(f"|FINISHED|: {input_dir} TO {output_dir}\n")

            # Write date string to the first line of each log file
            print(f"{GREEN}signal_to_text execution complete.{NC}")
    except subprocess.CalledProcessError as e:
        print(f"Error during C execution: {e}")
        sys.exit(e.returncode)

    try:   
        print(f"{YELLOW}Running signal_to_psd...{NC}")
        cmd = ["python3", "./py/signal_to_psd.py", "--file-type", mode, "-d", output_dir, "-l", py_log_file_path]
        with open(py_log_file_path, "w") as py_log_file:
            py_log_file.write(f"Python Log started on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} with cmd: {cmd}\n")
        # subprocess.run(cmd, check=True)
        process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        process.wait()

        print(f"{GREEN}signal_to_psd execution complete.{NC}")
    except subprocess.CalledProcessError as e:
        print(f"Error during py execution: {e}")
        sys.exit(e.returncode)

    # Calculate execution time
    end_time = time.time()
    execution_time = int(end_time - start_time)
    print(f"FINE! {BLUE}Execution completed in {execution_time} seconds.{NC}")