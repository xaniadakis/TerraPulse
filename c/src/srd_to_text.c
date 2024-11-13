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
#define VERSION "1.0"  // Specify the version here

const char *INPUT_DIR = "/mnt/e/KalpakiSortedData/160607_2";
const char *OUTPUT_DIR = "/mnt/c/Users/shumann/Documents/GaioPulse/srd_output";
size_t OUTPUT_DIR_PATH_LEN = 0;
size_t OUTPUT_FILE_PATH_LEN = 0;
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
        fprintf(file, "Data Origin: Greek\n"); // Additional line indicating data origin
        fprintf(file, "Total Files Written in Output Directory: %d\n", file_count);
        fclose(file);
    } else {
        fprintf(stderr, "Error creating metadata file in: %s\n", output_dir_path);
    }
}

void process_file(const char *file_path) {

    SrdData data = read_srd_file(file_path);

    if (!data.ok) {
        printf("Failed to load data.\n");
        return;
    } 
    // else {
    //     char date_str[30];
    //     format_date(data.date, date_str, sizeof(date_str));
    //     printf("Date: %s\n", date_str);
    //     printf("Sample Rate: %lf\n", data.fs);
    //     printf("Number of samples: %d\n", data.N);
    //     printf("Channel Count: %s\n", data.ch == 0 ? "Single" : "Dual");
    //     printf("Battery Voltage: %e V\n", data.vbat);
    // }

    int downsample_factor = (int)(data.fs / 100);  // Calculate downsample factor for 100 Hz
    
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
    snprintf(output_path, OUTPUT_FILE_PATH_LEN, "%s/%s.txt", dir_path, date_str);
    downsample_srd_signal(data, downsample_factor, output_path);

    // Free the allocated memory
    free(dir_path); 
    free(output_path);
    free(date_str); 
    free(data.x);
    if (data.ch == 1) free(data.y);

    current_dir_file_count++;  // Increment file count for the current directory
    processed_files++;

    // if (processed_files % 100 == 0) 
    print_progress(processed_files, total_files);
}

void count_files(const char *dir_path) {
    struct dirent *entry;
    DIR *dir = opendir(dir_path);
    char path[1024];

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) {
            snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
            count_files(path);
        } else if (strstr(entry->d_name, ".SRD")) {
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
        } else if (strstr(entry->d_name, ".SRD")) {
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
    OUTPUT_DIR_PATH_LEN = strlen(OUTPUT_DIR) + strlen("YYYYMMDD") + 2;
    OUTPUT_FILE_PATH_LEN = OUTPUT_DIR_PATH_LEN + strlen("YYYYMMDDHHMMSS.txt") + 2;
    start_time = time(NULL);
    traverse_directory(INPUT_DIR);

    // Create metadata for the last directory processed
    if (current_output_dir[0] != '\0') {
        create_metadata_file(current_output_dir, current_dir_file_count);
    }

    printf("\nProcessing complete.\n");
    return 0;
}
