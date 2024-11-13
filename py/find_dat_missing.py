import os
from collections import defaultdict

def find_missing_dat_files(parent_dir, log_file_path="missing_files.log"):
    missing_files_count = defaultdict(lambda: {"missing": 288, "found_files": set(), "path": None})
    good = 0
    bad = 0
    
    for root, dirs, files in os.walk(parent_dir):
        for subdir in dirs:
            if subdir.isdigit() and len(subdir) == 8:
                subdir_path = os.path.join(root, subdir)
                
                expected_dat_files = {f"{subdir}{hour:02d}{minute:02d}.dat" for hour in range(24) for minute in range(0, 60, 5)}
                actual_files = set(os.listdir(subdir_path))
                
                # Update found files with files from the current directory occurrence
                present_dat_files = {file for file in actual_files if file in expected_dat_files}
                missing_files_count[subdir]["found_files"].update(present_dat_files)
                
                # Update path if this is the first time encountering this subdir or keep the most complete path
                if missing_files_count[subdir]["path"] is None:
                    missing_files_count[subdir]["path"] = subdir_path

    # Calculate missing files for each date by comparing the full set of expected files with the combined found files
    sorted_missing_files_count = dict(sorted(missing_files_count.items()))
    
    # Write results to the log file and print summary
    with open(log_file_path, "w") as log_file:
        for subdir, info in sorted_missing_files_count.items():
            expected_dat_files = {f"{subdir}{hour:02d}{minute:02d}.dat" for hour in range(24) for minute in range(0, 60, 5)}
            missing_files = expected_dat_files - info["found_files"]
            missing_count = len(missing_files)
            
            if missing_count == 0:
                good += 1
            else:
                bad += 1
                print(f"{bad}. {subdir} dir - Missing dat files: {missing_count}")
                log_file.write(f"Missing files in {subdir} (path: {info['path']}):\n")
                for file in sorted(missing_files):
                    log_file.write(f" - {file}\n")
    
    print(f"{good} good days & {bad} bad days.")

# Example usage:
parent_directory = "/mnt/e/POLISH_DATA/Raw_Data"
find_missing_dat_files(parent_directory)
