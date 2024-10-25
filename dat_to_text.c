#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <fftw3.h>
#include "feature.h"
#include "io.h"
#include "signanalysis.h"
#include <stdbool.h>
#include <ftw.h>
#include <libgen.h> // Required for dirname
#include <sys/types.h> // Required for mkdir
#include <unistd.h> // Required for access

#define PROGRESS_BAR_WIDTH 100  // Width of the progress bar
#define TEST 0

// ANSI escape codes for color
#define COLOR_RESET "\033[0m"
#define COLOR_FILLED "\033[32m"  // Green for filled part
#define COLOR_EMPTY "\033[37m"   // White for empty part
#define COLOR_RED "\033[91m"   // White for empty part

void print_progress_bar(int progress, int total) {
    int bar_width = (progress * PROGRESS_BAR_WIDTH) / total; // Calculate filled width
    printf("\r|");
    for (int i = 0; i < PROGRESS_BAR_WIDTH; i++) {
        if (i < bar_width) {
            printf(COLOR_FILLED "█" COLOR_RESET);  // Filled part using a block character
        } else {
            printf(COLOR_RED "░" COLOR_RESET);    // Empty part using a light block character
        }
    }
    // printf("| %.4f%%", (progress / (double)total) * 100); // Display percentage
    printf("| %d/%d", progress, total); 
    fflush(stdout); // Ensure immediate output
}

// Global variables for output directory and file count
const char *output_dir = "./output/";
int total_files = 0;
int processed_files = 0;

int process_file(const char *input_file, const char *input_dir) {
    int *HNS = NULL, *HEW = NULL, nr = 0;
    if (read_dat_file(input_file, &HNS, &HEW, &nr) != 0) {
        printf("Something went wrong while reading file: %s\n", input_file);
        return 1;
    }

    double *calibrated_HNS = NULL, *calibrated_HEW = NULL;
    calibrate_HYL(HNS, HEW, nr, &calibrated_HNS, &calibrated_HEW);

    // Get relative path by removing input_dir prefix
    const char *relative_path = input_file + strlen(input_dir);
    char output_file_path[1028];
    snprintf(output_file_path, sizeof(output_file_path), "%s%s", output_dir, relative_path);

    // Remove .dat extension if it exists
    char *dat_extension = strstr(output_file_path, ".dat");
    if (dat_extension) {
        *dat_extension = '\0';  // Truncate the string at the .dat extension
    }

    // Create necessary subdirectories in the output path
    char *dir_path = strdup(output_file_path);
    char *sub_dir = dirname(dir_path);
    struct stat st = {0};

    // Create intermediate directories if they don't exist
    char temp_path[1028] = "";
    for (char *p = strtok(sub_dir, "/"); p; p = strtok(NULL, "/")) {
        strcat(temp_path, p);
        strcat(temp_path, "/");
        if (stat(temp_path, &st) == -1) {
            mkdir(temp_path, 0700);
        }
    }
    free(dir_path);

    // Create output file name with .txt extension
    char output_file[1032];
    snprintf(output_file, sizeof(output_file), "%s.txt", output_file_path);

    int downsampled_length = nr / DOWNSAMPLING_FACTOR;
    double *downsampled_HNS = (double *) malloc(downsampled_length * sizeof(double));
    double *downsampled_HEW = (double *) malloc(downsampled_length * sizeof(double));

    downsample_signal(calibrated_HNS, downsampled_HNS, downsampled_length);
    downsample_signal(calibrated_HEW, downsampled_HEW, downsampled_length);

    save_signals(downsampled_HNS, downsampled_HEW, NULL, NULL, NULL, downsampled_length, output_file);

    free(HNS);
    free(HEW);
    free(calibrated_HNS);
    free(calibrated_HEW);
    free(downsampled_HNS);
    free(downsampled_HEW);

    processed_files++;
    print_progress_bar(processed_files, total_files);
    return 0;
}

// Recursive function to traverse directories and count files
void count_files(const char *dir_path) {
    DIR *dir = opendir(dir_path);
    if (!dir) {
        perror("Failed to open directory");
        return;
    }

    struct dirent *entry;
    char path[1024];

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR) {
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
            count_files(path);  // Recurse into the subdirectory
        } else if (strstr(entry->d_name, ".dat")) {
            total_files++;  // Count .dat files
        }
    }

    closedir(dir);
}

// Recursive function to traverse directories and process files
void traverse_directory(const char *dir_path, const char *input_dir) {
    DIR *dir = opendir(dir_path);
    if (!dir) {
        perror("Failed to open directory");
        return;
    }

    struct dirent *entry;
    char path[1024];

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR) {
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
            traverse_directory(path, input_dir);  // Recurse into the subdirectory
        } else if (strstr(entry->d_name, ".dat")) {
            snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
            process_file(path, input_dir);  // Pass input_dir to process_file
        }
    }

    closedir(dir);
}

int main() {
    clock_t total_start = clock();
    const char *input_dir = "/mnt/e/POLISH_DATA/Raw_Data/Pol_Sr_dataMay22";

    struct stat st = {0};
    if (stat(output_dir, &st) == -1) {
        mkdir(output_dir, 0700);
    }

    count_files(input_dir);
    if (total_files == 0) {
        printf("No .dat files found.\n");
        return 0;
    }
    printf("Total files are: %d\n", total_files);

    traverse_directory(input_dir, input_dir);

    clock_t total_end = clock();
    double total_time = (double)(total_end - total_start) / CLOCKS_PER_SEC;
    printf("\nTotal time for the whole process: %.4f seconds\n", total_time);

    return 0;
}
