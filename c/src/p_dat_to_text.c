#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <libgen.h>
#include "feature.h"
#include "io.h"
#include "signanalysis.h"

#define PROGRESS_BAR_WIDTH 100
#define DOWNSAMPLING_FACTOR 30

// ANSI escape codes for color
#define COLOR_RESET "\033[0m"
#define COLOR_FILLED "\033[32m"  // Green for filled part
#define COLOR_EMPTY "\033[37m"   // White for empty part
#define COLOR_RED "\033[91m"   // White for empty part

pthread_mutex_t progress_mutex = PTHREAD_MUTEX_INITIALIZER;

const char *output_dir = "/mnt/c/Users/shumann/Documents/GaioPulse/output";
int total_files = 0;
int processed_files = 0;

typedef struct {
    const char *input_file;
    const char *input_dir;
} FileTask;

void print_progress_bar(int progress, int total) {
    int bar_width = (progress * PROGRESS_BAR_WIDTH) / total;
    printf("\r|");
    for (int i = 0; i < PROGRESS_BAR_WIDTH; i++) {
        if (i < bar_width) printf(COLOR_FILLED "█" COLOR_RESET);
        else printf(COLOR_RED "░" COLOR_RESET);
    }
    printf("| %d/%d", progress, total);
    fflush(stdout);
}

void create_output_directory(const char *path) {
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0700); // create intermediate directories
            *p = '/';
        }
    }
    mkdir(tmp, 0700); // create final directory if it doesn't exist
}

int process_file(const char *input_file, const char *input_dir) {
    int *HNS = NULL, *HEW = NULL, nr = 0;
    if (read_dat_file(input_file, &HNS, &HEW, &nr) != 0) {
        printf("Something went wrong while reading file: %s\n", input_file);
        return 1;
    }

    double *calibrated_HNS = NULL, *calibrated_HEW = NULL;
    calibrate_HYL(HNS, HEW, nr, &calibrated_HNS, &calibrated_HEW);

    const char *relative_path = input_file + strlen(input_dir);
    if (relative_path[0] == '/' || relative_path[0] == '\\') {
        relative_path++;
    }

    char date_dir[1024];
    strncpy(date_dir, relative_path, sizeof(date_dir) - 1);
    date_dir[sizeof(date_dir) - 1] = '\0';
    char *slash_pos = strchr(date_dir, '/');
    if (slash_pos) {
        *slash_pos = '\0';
    }

    char output_dir_path[1028];
    snprintf(output_dir_path, sizeof(output_dir_path), "%s/%s", output_dir, date_dir);

    // Ensure all intermediate directories in the output path are created
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", output_dir_path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0700);  // create intermediate directories
            *p = '/';
        }
    }
    mkdir(tmp, 0700);  // create the final directory if it doesn't exist

    char output_file[1032];
    snprintf(output_file, sizeof(output_file), "%s/%s", output_dir_path, basename(strdup(input_file)));

    // Replace .dat extension with .txt
    char *dat_extension = strstr(output_file, ".dat");
    if (dat_extension) {
        strcpy(dat_extension, ".txt");
    }

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

    return 0;
}

void *thread_process_file(void *arg) {
    FileTask *task = (FileTask *)arg;
    process_file(task->input_file, task->input_dir);

    pthread_mutex_lock(&progress_mutex);
    processed_files++;
    if (processed_files % 200 == 0 || processed_files == total_files) {  // Update progress every 10 files
        print_progress_bar(processed_files, total_files);
    }
    pthread_mutex_unlock(&progress_mutex);

    free((void *)task->input_file);
    free(task);
    return NULL;
}

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
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
            snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
            count_files(path);
        } else if (strstr(entry->d_name, ".dat")) {
            total_files++;
        }
    }
    closedir(dir);
}

void traverse_directory(const char *dir_path, const char *input_dir, pthread_t *threads, int max_threads) {
    DIR *dir = opendir(dir_path);
    if (!dir) return;

    struct dirent *entry;
    char path[1024];
    int thread_count = 0;

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR) {
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
            snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
            traverse_directory(path, input_dir, threads, max_threads);
        } else if (strstr(entry->d_name, ".dat")) {
            snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);

            FileTask *task = malloc(sizeof(FileTask));
            if (task == NULL) {
                perror("Failed to allocate memory for task");
                continue;
            }
            task->input_file = strdup(path);
            task->input_dir = input_dir;

            if (pthread_create(&threads[thread_count++], NULL, thread_process_file, task) != 0) {
                perror("Failed to create thread");
                free(task);
            }

            if (thread_count >= max_threads) {
                for (int i = 0; i < thread_count; i++) {
                    pthread_join(threads[i], NULL);
                }
                thread_count = 0;
            }
        }
    }
    closedir(dir);

    for (int i = 0; i < thread_count; i++) {
        pthread_join(threads[i], NULL);
    }
}

int main() {
    const char *input_dir = "/mnt/e/POLISH_DATA/Raw_Data/Polish_Data_winter_22";
    printf("Counting files...\n");
    count_files(input_dir);
    if (total_files == 0) {
        printf("No .dat files found.\n");
        return 0;
    }
    printf("Total files are: %d\n", total_files);

    int max_threads = sysconf(_SC_NPROCESSORS_ONLN);
    if (max_threads < 1) max_threads = 1;
    printf("Using %d out of %d threads.\n", max_threads - 2, max_threads);

    clock_t start = clock();
    pthread_t threads[max_threads];
    traverse_directory(input_dir, input_dir, threads, max_threads - 2);
    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\nProcessing complete. Time taken: %f seconds\n", time_taken);
    return 0;
}
