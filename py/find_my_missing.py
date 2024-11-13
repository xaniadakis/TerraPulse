import os

def check_missing_files(parent_dir, log_file_path="missing_files.log"):
    missing_files_count = {}
    bad = 0
    good = 0
    with open(log_file_path, "w") as log_file:
        for subdir in os.listdir(parent_dir):
            subdir_path = os.path.join(parent_dir, subdir)
            
            if os.path.isdir(subdir_path) and subdir.isdigit() and len(subdir) == 8:
                expected_txt_files = {f"{subdir}{hour:02d}{minute:02d}.txt" for hour in range(24) for minute in range(0, 60, 5)}
                expected_zst_files = {f"{subdir}{hour:02d}{minute:02d}.zst" for hour in range(24) for minute in range(0, 60, 5)}
                
                actual_files = set(os.listdir(subdir_path))
                
                missing_txt = expected_txt_files - actual_files
                missing_zst = expected_zst_files - actual_files
                
                if missing_txt or missing_zst:
                    missing_files_count[subdir] = {
                        "txt": len(missing_txt),
                        "zst": len(missing_zst)
                    }
                    
                    log_file.write(f"Missing files in {subdir}:\n")
                    for file in sorted(missing_txt | missing_zst):
                        log_file.write(f" - {file}\n")
                else:
                    missing_files_count[subdir] = "OK"

    # Summary of missing files by directory

    for subdir, status in missing_files_count.items():
        if status == "OK":
            good += 1
            # print(f"Directory {subdir} is OK, no missing files.")
        else:
            bad += 1
            print(f"{bad}. {subdir} dir - Missing txt files: {status['txt']}, Missing zst files: {status['zst']}")
    print(f"{good} good days & {bad} bad days.")
    
# Example usage:
parent_directory = "../output"
check_missing_files(parent_directory)
