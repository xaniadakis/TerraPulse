#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include "feature.h"
#include "io.h"
#include "signanalysis.h"
#include <libgen.h>
#include <errno.h>
#include "log.h"

#define PROGRESS_BAR_WIDTH 50
#define DOWNSAMPLING_FACTOR 30
#define BAR_UPDATE_FREQUENCY 5
#define VERSION "1.0"  // Specify the version here
#define DEBUG 0

// File extensions
#define POLSKI_LOGGER_FILE_TYPE ".dat"
#define POLSKI_FRESH_FILE_TYPE ".pol"
#define HELLENIC_LOGGER_FILE_TYPE ".SRD"
#define HELLENIC_FRESH_FILE_TYPE ".hel"

FILE *log_file = NULL;  // Definition

// Mode enum
typedef enum { HELLENIC_LOGGER, POLSKI_LOGGER } Mode;
Mode current_mode;

// const char *INPUT_DIR = NULL;
int NUM_INPUT_DIRS = 0;
const char **INPUT_DIRS = NULL;
const char *OUTPUT_DIR = NULL;
int total_files = 0, processed_files = 0;
time_t start_time;

// Variables to keep track of the current output directory and file count in it
char current_output_dir[1028] = "";
int current_dir_file_count = 0;
size_t OUTPUT_DIR_PATH_LEN = 0;
size_t OUTPUT_FILE_PATH_LEN = 0;

#include <stdio.h>
#include <time.h>

#define PROGRESS_BAR_WIDTH 50

extern time_t start_time;  // Assuming start_time is declared elsewhere

void print_progress(int progress, int total, int simple_mode) {
    time_t current_time = time(NULL);
    double elapsed = difftime(current_time, start_time);
    double remaining = (elapsed / progress) * (total - progress);

    if (remaining < 0 || remaining > 999999 || progress < 70) remaining = 0;  // Sanity check

    if (simple_mode) {
        // Simple mode: Print only a single line with progress info
        printf("\rDone: %d/%d | Elapsed: %02d:%02d:%02d | Left: %02d:%02d:%02d",
               progress, total,
               (int)(elapsed / 3600), (int)(elapsed / 60) % 60, (int)elapsed % 60,  // Elapsed (hh:mm:ss)
               (int)(remaining / 3600), (int)(remaining / 60) % 60, (int)remaining % 60);  // Remaining (hh:mm:ss)
    } else {
        // Normal mode: Show progress bar
        int bar_width = (progress * PROGRESS_BAR_WIDTH) / total;

        printf("\r|");
        for (int i = 0; i < PROGRESS_BAR_WIDTH; i++)
            printf(i < bar_width ? "\033[32m█\033[0m" : "\033[91m░\033[0m");

        printf("| %d/%d | Elapsed: %02d:%02d:%02d | Left: %02d:%02d:%02d", 
               progress, total, 
               (int)(elapsed / 3600), (int)(elapsed / 60) % 60, (int)elapsed % 60,  // Elapsed (hh:mm:ss)
               (int)(remaining / 3600), (int)(remaining / 60) % 60, (int)remaining % 60);  // Remaining (hh:mm:ss)
    }

    fflush(stdout);
}

void create_dir(const char *path) {
    char temp[1024];
    snprintf(temp, sizeof(temp), "%s", path);
    for (char *p = temp + 1; *p; p++) {
        if (*p == '/') { *p = '\0'; mkdir(temp, 0700); *p = '/'; }
    }
    mkdir(temp, 0700);
}

void create_metadata_file(const char *output_dir_path, int file_count) {
    char metadata_file_path[1050];
    snprintf(metadata_file_path, sizeof(metadata_file_path), "%s/metadata", output_dir_path);

    FILE *file = fopen(metadata_file_path, "w");
    if (file) {
        fprintf(file, "Version: %s\n", VERSION);
        fprintf(file, "Data Origin: Polish\n"); // Additional line indicating data origin
        fprintf(file, "Total Files Written in Output Directory: %d\n", file_count);
        fclose(file);
    } else {
        fprintf(stderr, "Error creating metadata file in: %s\n", output_dir_path);
    }
}

void process_dat_file(const char *file_path) {
    int *HNS = NULL, *HEW = NULL, nr = 0;
    if (read_dat_file(file_path, &HNS, &HEW, &nr)) {
        fprintf(stderr, "Error reading file: %s\n", file_path);
        return;
    }
    if(nr==0){
        printf("\nBad input file: %s\n", file_path);
        total_files--;
        free(HNS); 
        free(HEW);
        return;
    }

    const char* date_dir = strndup(strrchr(file_path, '/') + 1, 8);

    double *calibrated_HNS = NULL, *calibrated_HEW = NULL;
    calibrate_HYL(HNS, HEW, nr, date_dir, &calibrated_HNS, &calibrated_HEW);

    char output_dir_path[1028];
    snprintf(output_dir_path, sizeof(output_dir_path), "%s/%s", OUTPUT_DIR, date_dir);
    create_dir(output_dir_path);

    // Check if we're still in the same directory, if not, create the metadata for the last one
    if (strcmp(current_output_dir, output_dir_path) != 0) {
        if (current_output_dir[0] != '\0') {
            create_metadata_file(current_output_dir, current_dir_file_count);
        }
        // Update the current directory and reset file count
        strncpy(current_output_dir, output_dir_path, sizeof(current_output_dir));
        current_dir_file_count = 0;
    }

    char *file_path_copy = strdup(file_path);
    char *base_name = basename(file_path_copy);
    base_name[strlen(base_name) - 4] = '\0'; // Remove ".dat"

    char output_file[1033];
    int ret = snprintf(output_file, sizeof(output_file), "%s/%s%s", output_dir_path, base_name, POLSKI_FRESH_FILE_TYPE);

    // printf("%s\n", base_name);

    if (ret >= sizeof(output_file)) {
        fprintf(stderr, "Warning: output file path truncated for %s\n", file_path);
        output_file[sizeof(output_file) - 1] = '\0'; // Ensure null-termination
    }

    int downsampled_length = nr / DOWNSAMPLING_FACTOR;
    double *downsampled_HNS = malloc(downsampled_length * sizeof(double));
    double *downsampled_HEW = malloc(downsampled_length * sizeof(double));

    downsample_dat_signal(calibrated_HNS, downsampled_HNS, downsampled_length);
    downsample_dat_signal(calibrated_HEW, downsampled_HEW, downsampled_length);
    save_signals(downsampled_HNS, downsampled_HEW, NULL, NULL, NULL, downsampled_length, output_file);

    free(HNS); free(HEW); free(calibrated_HNS); free(calibrated_HEW);
    free(downsampled_HNS); free(downsampled_HEW);
    free(file_path_copy); free((void*)date_dir);

    current_dir_file_count++;  // Increment file count for the current directory
    processed_files++;

    if (processed_files % BAR_UPDATE_FREQUENCY == 0) print_progress(processed_files, total_files, 1);
}

void process_srd_file(const char *file_path) {

    SrdData data = read_srd_file(file_path);

    if (!data.ok) {
        printf("Failed to load data.\n");
        total_files--;
        return;
    } else if(DEBUG) {
        char date_str[30];
        format_date(data.date, date_str, sizeof(date_str));
        printf("Date: %s | Sample Rate: %.2lf Hz | Samples: %d | Channels: %s | Battery: %.2e V\n",
            date_str, data.fs, data.N, data.ch == 0 ? "Single" : "Dual", data.vbat);
    }

    // Determine per-channel sample count
    int samples_per_channel = (data.ch == 1) ? (data.N / 2) : data.N;
    // Calculate initial downsample factor for target 100 Hz
    int downsample_factor = (int)(data.fs / 100);
    // Adjust downsample factor to ensure integer result
    // while (samples_per_channel % downsample_factor != 0) {
    //     downsample_factor--;
    // }
    if(DEBUG){
        int downsampled_samples_per_channel = samples_per_channel / downsample_factor;
        int total_downsampled_samples = (data.ch == 1) ? (downsampled_samples_per_channel * 2) : downsampled_samples_per_channel;
        printf("Downsample Factor: %d, Downsampled Samples Per Channel: %d, Total Downsampled Samples: %d\n",
                downsample_factor, downsampled_samples_per_channel, total_downsampled_samples);
    }

    char *output_path = malloc(OUTPUT_FILE_PATH_LEN);
    if (output_path == NULL) {
        printf("Output path malloc failed.\n");
        return;
    }

    char *date_str = get_filename(data.date);
    if (date_str == NULL) {
        printf("Output filename malloc failed.\n");
        free(output_path); 
        return;
    }

    char *dir_path = malloc(OUTPUT_DIR_PATH_LEN);
    if (dir_path == NULL) {
        printf("Output path malloc failed.\n");
        free(output_path); 
        free(date_str); 
        return;
    }
    snprintf(dir_path, OUTPUT_DIR_PATH_LEN, "%s/%.8s", OUTPUT_DIR, date_str);
    create_dir(dir_path); // Create the directory

    // Append the filename to the directory path
    snprintf(output_path, OUTPUT_FILE_PATH_LEN, "%s/%s%s", dir_path, date_str, HELLENIC_FRESH_FILE_TYPE);
    downsample_srd_signal(data, downsample_factor, output_path);

    // Free the allocated memory
    free(dir_path); 
    free(output_path);
    free(date_str); 
    free(data.x);
    if (data.ch == 1) free(data.y);

    current_dir_file_count++;  // Increment file count for the current directory
    processed_files++;

    if (processed_files % BAR_UPDATE_FREQUENCY == 0) print_progress(processed_files, total_files, 1);
}

void count_files(const char *dir_path, const char *file_type) {
    struct dirent *entry;
    DIR *dir = opendir(dir_path);
    char path[1024];

    if (strstr(dir_path, "DONT_PROCESS") != NULL) {
        printf("Skipping directory: '%s' while counting\n", dir_path);  // Print the directory name before returning
        return;  // Ignore directory if name contains "DONT_PROCESS"
    }

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) {
            snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
            count_files(path, file_type);
        } else {
            if (strcasestr(entry->d_name, file_type)) {
                total_files++;
            }
        }
    }
    closedir(dir);
}

void traverse_directory(const char *dir_path) {
    struct dirent *entry;
    DIR *dir = opendir(dir_path);
    char path[1024];

    if (strstr(dir_path, "DONT_PROCESS") != NULL) {
        printf("Skipping directory: '%s' while processing.\n", dir_path);  // Print the directory name before returning
        return;  // Ignore directory if name contains "DONT_PROCESS"
    }

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) {
            snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
            traverse_directory(path);
        } else {
            const char *file_type = current_mode == HELLENIC_LOGGER ? HELLENIC_LOGGER_FILE_TYPE : POLSKI_LOGGER_FILE_TYPE;
            if (strcasestr(entry->d_name, file_type)) {
                snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
                if (current_mode == HELLENIC_LOGGER) {
                    process_srd_file(path);
                } else if (current_mode == POLSKI_LOGGER) {
                    process_dat_file(path);
                } else {
                    return;
                }
            }
        }
    }
    print_progress(processed_files, total_files, 1);
    closedir(dir);
}

int main(int argc, char *argv[]) {
    // if (argc < 4) {
    //     fprintf(stderr, "Usage: %s <mode: pol|hel> <input_dir1> <input_dir2> ... <input_dirN> <output_dir>\n", argv[0]);
    //     return 1;
    // }
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <mode: pol|hel> <log_file> <input_dir1> ... <input_dirN> <output_dir>\n", argv[0]);
        return 1;
    }
    start_time = time(NULL);
    log_file = fopen(argv[2], "a");
    if (!log_file) {
        fprintf(stderr, "Error: Could not open log file %s for writing.\n", argv[2]);
        return 1;
    }
    printf("C started logging...\n");

    const char *mode_arg = argv[1];
    NUM_INPUT_DIRS = argc - 4;
    INPUT_DIRS = (const char **)malloc(NUM_INPUT_DIRS * sizeof(char *));
    for (int i = 0; i < NUM_INPUT_DIRS; i++) {
        INPUT_DIRS[i] = argv[i + 3];
    }
    OUTPUT_DIR = argv[argc - 1]; // Last argument is the output directory
    // OUTPUT_DIR = argv[3];

    if (strcmp(mode_arg, "hel") == 0) {
        current_mode = HELLENIC_LOGGER;
    } else if (strcmp(mode_arg, "pol") == 0) {
        current_mode = POLSKI_LOGGER;
    } else {
        fprintf(stderr, "Invalid mode. Use 'pol' or 'hel'.\n");
        return 1;
    }
    printf("Mode: %s\n", current_mode == HELLENIC_LOGGER ? "Hellenic" : "Polski");
    const char *file_type = current_mode == HELLENIC_LOGGER ? HELLENIC_LOGGER_FILE_TYPE : POLSKI_LOGGER_FILE_TYPE;

    struct stat st;

    // Check output directory
    if (stat(OUTPUT_DIR, &st) != 0) {
        if (errno == ENOENT) {
            printf("Output Directory '%s' does not exist. Creating it...\n", OUTPUT_DIR);
            if (mkdir(OUTPUT_DIR, 0755) == 0) {
                printf("Output Directory successfully created.\n");
            } else {
                printf("Failed to create Output Directory '%s'. (errno: %d)\n", OUTPUT_DIR, errno);
                return 1;
            }
        } else {
            printf("Error: Could not access Output Directory '%s'. (errno: %d)\n", OUTPUT_DIR, errno);
            return 1;
        }
    } else if (!S_ISDIR(st.st_mode)) {
        printf("Error: Output Directory '%s' exists but is not a directory.\n", OUTPUT_DIR);
        return 1;
    }
    printf("Will read into '%s'.\n", OUTPUT_DIR);

    // First, count all files in all input directories
    for (int i = 0; i < NUM_INPUT_DIRS; i++) {
        const char *input_dir = INPUT_DIRS[i];

        // Check input directory
        if (stat(input_dir, &st) != 0) {
            printf("Error: Input Directory '%s' does not exist. (stat returned: %d)\n", input_dir, errno);
            continue; // Skip to the next directory
        } 
        if (!S_ISDIR(st.st_mode)) {
            printf("Error: Input Directory '%s' is not a directory.\n", input_dir);
            continue;
        }

        printf("\nCounting files in input directory: '%s'\n", input_dir);
        count_files(input_dir, file_type);
    }

    if (total_files == 0) {
        printf("No %s files found in any of the input directories.\n", file_type);
        free(INPUT_DIRS);
        return 0;
    }
    printf("Total files found in all input directories: %d\n", total_files);

    OUTPUT_DIR_PATH_LEN = strlen(OUTPUT_DIR) + strlen("YYYYMMDD") + 2;
    OUTPUT_FILE_PATH_LEN = OUTPUT_DIR_PATH_LEN + strlen("YYYYMMDDHHMMSS.txt") + 2;

    printf("We have %d input directories\n", NUM_INPUT_DIRS);
    for (int i = 0; i < NUM_INPUT_DIRS; i++) {
        const char *input_dir = INPUT_DIRS[i];
        printf("\nProcessing input directory: '%s'\n", input_dir);
        traverse_directory(input_dir);
        if (current_output_dir[0] != '\0') {
            create_metadata_file(current_output_dir, current_dir_file_count);
        }
    }

    printf("\nC Processing complete.\n");
    fclose(log_file);
    free(INPUT_DIRS);
    return 0;
}