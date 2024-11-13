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

#define PROGRESS_BAR_WIDTH 80
#define DOWNSAMPLING_FACTOR 30
#define BAR_UPDATE_FREQUENCY 5
#define VERSION "1.0"  // Specify the version here

const char *INPUT_DIR = "/mnt/e/POLISH_DATA/Raw_Data/POLDATA Mayjuly22/20220706";
const char *OUTPUT_DIR = "/mnt/c/Users/shumann/Documents/GaioPulse/test_output";
int total_files = 0, processed_files = 0;
time_t start_time;

// Variables to keep track of the current output directory and file count in it
char current_output_dir[1028] = "";
int current_dir_file_count = 0;

char* get_relative_path(const char* path) {
    const char* relative_path = path + strlen(INPUT_DIR);
    const char* last_slash = strrchr(relative_path, '/');
    return last_slash ? strndup(relative_path, last_slash - relative_path) : strdup(relative_path);
}

void print_progress(int progress, int total) {
    int bar_width = (progress * PROGRESS_BAR_WIDTH) / total;
    time_t current_time = time(NULL);
    double elapsed = difftime(current_time, start_time);
    double remaining = (elapsed / progress) * (total - progress);

    printf("\r|");
    for (int i = 0; i < PROGRESS_BAR_WIDTH; i++)
        printf(i < bar_width ? "\033[32m█\033[0m" : "\033[91m░\033[0m");
    
    printf("| %d/%d | Elapsed: %02d:%02d | Left: %02d:%02d", 
           progress, total, (int)(elapsed/60), (int)elapsed % 60, (int)(remaining/60), (int)remaining % 60);
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

void process_file(const char *file_path) {
    int *HNS = NULL, *HEW = NULL, nr = 0;
    if (read_dat_file(file_path, &HNS, &HEW, &nr)) {
        fprintf(stderr, "Error reading file: %s\n", file_path);
        return;
    }
    if(nr==0){
        printf("\nBad input file: %s\n", file_path);
        free(HNS); 
        free(HEW);
        return;
    }
    // if(nr<900000){
    //     struct stat st;
    //     if (stat(file_path, &st) == 0) {
    //         printf("\nFile: %s, %d samples, Size: %ld bytes\n", file_path, nr, st.st_size);
    //     } else {
    //         printf("\nFile: %s, %d samples, size=error\n", file_path, nr);
    //     }
    // }
    double *calibrated_HNS = NULL, *calibrated_HEW = NULL;
    calibrate_HYL(HNS, HEW, nr, &calibrated_HNS, &calibrated_HEW);

    const char* date_dir = strndup(strrchr(file_path, '/') + 1, 8);
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

    char output_file[1032];
    int ret = snprintf(output_file, sizeof(output_file), "%s/%s.txt", output_dir_path, base_name);

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

    if (processed_files % BAR_UPDATE_FREQUENCY == 0) print_progress(processed_files, total_files);
}

void count_files(const char *dir_path) {
    struct dirent *entry;
    DIR *dir = opendir(dir_path);
    char path[1024];

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) {
            snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
            count_files(path);
        } else if (strstr(entry->d_name, ".dat")) {
            total_files++;
        }
    }
    closedir(dir);
}

void traverse_directory(const char *dir_path) {
    struct dirent *entry;
    DIR *dir = opendir(dir_path);
    char path[1024];

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) {
            snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
            traverse_directory(path);
        } else if (strstr(entry->d_name, ".dat")) {
            snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
            process_file(path);
        }
    }
    closedir(dir);
}

int main() {
    printf("Counting files...\n");
    count_files(INPUT_DIR);
    if (total_files == 0) {
        printf("No .dat files found.\n");
        return 0;
    }
    printf("Total files: %d\n", total_files);

    start_time = time(NULL);
    traverse_directory(INPUT_DIR);

    // Create metadata for the last directory processed
    if (current_output_dir[0] != '\0') {
        create_metadata_file(current_output_dir, current_dir_file_count);
    }

    printf("\nProcessing complete.\n");
    return 0;
}
