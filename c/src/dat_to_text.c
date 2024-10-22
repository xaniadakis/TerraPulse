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

// Function to process a single file
int process_file(const char *input_file) {

    // printf("will read dat file\n");
    int *HNS = NULL, *HEW = NULL, nr = 0;
    if(read_dat_file(input_file, &HNS, &HEW, &nr)!=0){
        printf("Something went wrong while reading file: %s\n", input_file);
        return 1;
    }

    // printf("will calibrate_HYL\n");
    double *calibrated_HNS = NULL, *calibrated_HEW = NULL;
    calibrate_HYL(HNS, HEW, nr, &calibrated_HNS, &calibrated_HEW);

    // Generate output file name
    char output_file[1024];
    snprintf(output_file, sizeof(output_file), "%s%s.txt", output_dir, strtok(strrchr(input_file, '/') + 1, "."));
    
    TimeDomainFeatures* hns_features = NULL;
    // extract_time_domain_features(calibrated_HNS, nr);
    TimeDomainFeatures* hew_features = NULL;
    // extract_time_domain_features(calibrated_HEW, nr);

    // Call the PSD computation function
    // char psd_output_file[1024];
    // snprintf(psd_output_file, sizeof(psd_output_file), "%s%s_psd.txt", output_dir, strtok(entry->d_name, "."));
    // compute_psd(calibrated_HNS, nr, psd_output_file, sampling_frequency, min_freq, max_freq);

    // Allocate memory for frequencies and PSD values
    // int num_psd_points = nr / 2 + 1;
    // double *frequencies = (double *) malloc(sizeof(double) * num_psd_points);
    // double *psd_values = (double *) malloc(sizeof(double) * num_psd_points);

    // Compute the PSD
    // compute_psd(calibrated_HNS, nr, frequencies, psd_values, psd_output_file, sampling_frequency, min_freq, max_freq);

    // find_modes(downsampled_HNS, downsampled_length, sampling_frequency, min_freq, max_freq, 7); // Call with HNS signal
    // Harmonic* harmonics = analyze_schumann_harmonics(calibrated_HNS, nr, sampling_frequency);


    int downsampled_length = nr / DOWNSAMPLING_FACTOR;
    double *downsampled_HNS = (double *) malloc(downsampled_length * sizeof(double));
    double *downsampled_HEW = (double *) malloc(downsampled_length * sizeof(double));


    // printf("will downsample_signal\n");
    // Downsample the signals
    downsample_signal(calibrated_HNS, downsampled_HNS, downsampled_length);
    downsample_signal(calibrated_HEW, downsampled_HEW, downsampled_length);


    // printf("will save_signals\n");
    // Save downsampled signals to file
    save_signals(downsampled_HNS, downsampled_HEW, hns_features, hew_features, NULL, downsampled_length, output_file);


    // printf("will free allocated memory\n");
    // Free allocated memory
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
void traverse_directory(const char *dir_path) {
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
            traverse_directory(path);  // Recurse into the subdirectory
        } else if (strstr(entry->d_name, ".dat")) {
            snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
            process_file(path);  // Process the .dat file
        }
    }

    closedir(dir);
}

int main() {
    // Measure the total start time
    clock_t total_start = clock();

    const char *input_dir = "/mnt/e/POLISH_DATA/Raw_Data/070220_100320";

    // Create output directory if it doesn't exist
    struct stat st = {0};
    if (stat(output_dir, &st) == -1) {
        mkdir(output_dir, 0700);
    }

    // First pass: Count total files
    count_files(input_dir);
    if (total_files == 0) {
        printf("No .dat files found.\n");
        return 0;
    }
    printf("Total files are: %d\n", total_files);

    // Second pass: Process the files and show progress bar
    traverse_directory(input_dir);

    // Measure total end time
    clock_t total_end = clock();
    double total_time = (double)(total_end - total_start) / CLOCKS_PER_SEC;
    printf("\nTotal time for the whole process: %.4f seconds\n", total_time);

    return 0;
}