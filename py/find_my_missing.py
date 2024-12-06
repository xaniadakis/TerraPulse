import os
import argparse
from collections import defaultdict

def check_missing_files(generated_directory, original_directory, generated_file_type, original_file_type, log_file_path="missing_files.log"):
    gen_missing_files_details = {}
    gen_missing_files_count = {}
    gen_bad = 0
    gen_good = 0
    # with open(log_file_path, "w") as log_file:
    for subdir in os.listdir(generated_directory):
            subdir_path = os.path.join(generated_directory, subdir)
            
            if os.path.isdir(subdir_path) and subdir.isdigit() and len(subdir) == 8:
                expected_txt_files = {f"{subdir}{hour:02d}{minute:02d}.{generated_file_type}" for hour in range(24) for minute in range(0, 60, 5)}
                expected_zst_files = {f"{subdir}{hour:02d}{minute:02d}.zst" for hour in range(24) for minute in range(0, 60, 5)}
                
                actual_files = set(os.listdir(subdir_path))
                
                missing_txt = expected_txt_files - actual_files
                missing_zst = expected_zst_files - actual_files
                
                if missing_txt or missing_zst:
                    gen_missing_files_count[subdir] = {
                        generated_file_type: len(missing_txt),
                        "zst": len(missing_zst)
                    }
                    
                    filenames_without_extensions = [os.path.splitext(os.path.basename(path))[0] for path in missing_txt]
                    psdfiles_without_extensions = [os.path.splitext(os.path.basename(path))[0] for path in missing_zst]
                    gen_missing_files_details[subdir] = {
                        generated_file_type: sorted(filenames_without_extensions),
                        "zst": sorted(psdfiles_without_extensions),
                    }

                    # log_file.write(f"Missing files in {subdir}:\n")
                    # for file in sorted(missing_txt | missing_zst):
                    #     log_file.write(f" - {file}\n")
                else:
                    gen_missing_files_count[subdir] = "OK"

    # Summary of missing files by directory

    for subdir, status in gen_missing_files_count.items():
        if status == "OK":
            gen_good += 1
            # print(f"Directory {subdir} is OK, no missing files.")
        else:
            gen_bad += 1
            print(f"{gen_bad}. {subdir} dir - Missing {generated_file_type} files: {status[generated_file_type]}, Missing zst files: {status['zst']}")
    print(f"{gen_good} good days & {gen_bad} bad days.")

    og_missing_files_details = {}
    og_missing_files_count = defaultdict(lambda: {"missing": 288, "found_files": set(), "path": None})
    og_good = 0
    og_bad = 0
    
    for root, dirs, files in os.walk(original_directory):
        for subdir in dirs:
            if subdir.isdigit() and len(subdir) == 8:
                subdir_path = os.path.join(root, subdir)
                
                expected_dat_files = {f"{subdir}{hour:02d}{minute:02d}.{original_file_type}" for hour in range(24) for minute in range(0, 60, 5)}
                actual_files = set(os.listdir(subdir_path))
                
                # Update found files with files from the current directory occurrence
                present_dat_files = {file for file in actual_files if file in expected_dat_files}
                og_missing_files_count[subdir]["found_files"].update(present_dat_files)
                
                # Update path if this is the first time encountering this subdir or keep the most complete path
                if og_missing_files_count[subdir]["path"] is None:
                    og_missing_files_count[subdir]["path"] = subdir_path

    # Calculate missing files for each date by comparing the full set of expected files with the combined found files
    sorted_missing_files_count = dict(sorted(og_missing_files_count.items()))
    
    # Write results to the log file and print summary
    # with open(log_file_path, "w") as log_file:
    for subdir, info in sorted_missing_files_count.items():
        expected_dat_files = {f"{subdir}{hour:02d}{minute:02d}.{original_file_type}" for hour in range(24) for minute in range(0, 60, 5)}
        missing_files = expected_dat_files - info["found_files"]
        missing_count = len(missing_files)
        
        filenames_without_extensions = [os.path.splitext(os.path.basename(path))[0] for path in missing_files]
        og_missing_files_details[subdir] = {
            original_file_type: sorted(filenames_without_extensions),
        }
        
        if missing_count == 0:
            og_good += 1
        else:
            og_bad += 1
            print(f"{og_bad}. {subdir} dir - Missing {original_file_type} files: {missing_count}")
            # log_file.write(f"Missing files in {subdir} (path: {info['path']}):\n")
            # for file in sorted(missing_files):
            #     log_file.write(f" - {file}\n")
    print(f"{og_good} good days & {og_bad} bad days.")

    # Print the non generated files
    with open(log_file_path, "w") as log_file:
        for gen_subdir, gen_missing_files in gen_missing_files_details.items():
            # Check if the corresponding directory exists in og_missing_files_details
            if gen_subdir in og_missing_files_details:
                og_missing_files = og_missing_files_details[gen_subdir]

                # Remove elements from gen_details[generated_file_type] that exist in og_details[original_file_type]
                gen_missing_files[generated_file_type] = [
                    item for item in gen_missing_files[generated_file_type]
                    if item not in og_missing_files[original_file_type]
                ]
                gen_missing_files["zst"] = [
                    item for item in gen_missing_files["zst"]
                    if item not in og_missing_files[original_file_type]
                ]

                log_file.write(f"Missing files in {subdir}):\n")
                for i, file in enumerate(sorted(gen_missing_files[generated_file_type])):
                    log_file.write(f" {i}. {file}.{generated_file_type}\n")
                for i, file in enumerate(sorted(gen_missing_files["zst"])):
                    log_file.write(f" {i}. {file}.{"zst"}\n")
            else:
                print(f"OG DIR not found for GEN DIR: {gen_subdir}")

    # for og_subdir, og_missing_files in og_missing_files_details.items():
    #     if og_subdir in gen_missing_files_details:
    #         gen_missing_files = gen_missing_files_details[og_subdir]
    #     else:
    #         print(f"{og_subdir} is missing in original data but exists in generated data.")

    # for gen_subdir, gen_missing_files in gen_missing_files_details.items():
        # Check if the corresponding directory exists in og_missing_files_details
        if gen_subdir in og_missing_files_details:
            og_missing_files = og_missing_files_details[gen_subdir]

            # Get elements in og_missing_files[original_file_type] that are not in gen_missing_files[generated_file_type]
            missing_in_og_not_in_gen = [
                item for item in og_missing_files[original_file_type]
                if item not in gen_missing_files[generated_file_type]
            ]
            print(missing_in_og_not_in_gen)
            if(len(missing_in_og_not_in_gen[original_file_type])>0):
                print(f"Generated garbage in: {gen_subdir}! ({len(missing_in_og_not_in_gen[original_file_type])})")
        else:
            if(len(missing_in_og_not_in_gen[original_file_type])>0):
                print(f"Generated garbage in: {gen_subdir}!")


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

if __name__ == "__main__":
    original_file_type = None
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Search for missing files")
    parser.add_argument(
        "-t", "--file-type", 
        choices=['pol', 'hel'], 
        default='pol',
        help="Specify the file type to look for. Only 'pol' or 'hel' are allowed."
    )
    parser.add_argument(
        "-o", "--og-directory", 
        required=False,
        help="Specify the original directory."
    )
    parser.add_argument(
        "-g", "--gen-directory", 
        required=True,
        help="Specify the generated directory."
    )
    args = parser.parse_args()
    generated_directory = args.gen_directory
    original_directory = args.og_directory
    if args.file_type=="hel":
        original_file_type = "SRD"
    elif args.file_type=="pol":
        original_file_type = "dat"
    linux_generated_directory = translate_windows_to_linux_path(generated_directory)
    print(f"{args.file_type} Directory: {generated_directory} -> {linux_generated_directory}")  
    linux_original_directory = translate_windows_to_linux_path(original_directory)
    print(f"{original_file_type} Directory: {original_directory} -> {linux_original_directory}")  
    check_missing_files(linux_generated_directory, linux_original_directory, args.file_type, original_file_type)