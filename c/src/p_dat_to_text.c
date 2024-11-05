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

#define PROGRESS_BAR_WIDTH 80
#define DOWNSAMPLING_FACTOR 30
#define MAX_QUEUE_SIZE 1000
#define MAX_MISSING_FILES 288
#define MAX_DIRECTORIES 1024

pthread_mutex_t progress_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t queue_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t queue_cond = PTHREAD_COND_INITIALIZER;

const char *input_dir = "/mnt/e/POLISH_DATA/Raw_Data";
const char *output_dir = "/mnt/c/Users/shumann/Documents/GaioPulse/output";
int total_files = 0;
int processed_files = 0;
time_t start_time;

typedef struct {
    const char *input_file;
    const char *input_dir;
} FileTask;

typedef struct {
    int files_written;
    int files_read;
    int errors_occurred;
    const char *missing_files[MAX_MISSING_FILES];
    int missing_file_count;
} Metadata;

Metadata metadata_array[MAX_DIRECTORIES];
int metadata_index = 0;

FileTask *task_queue[MAX_QUEUE_SIZE];
int queue_front = 0;
int queue_back = 0;
int queue_count = 0;

void save_metadata(const char *output_dir_path, Metadata *metadata) {
    char metadata_file_path[1028];
    snprintf(metadata_file_path, sizeof(metadata_file_path), "%s/metadata.txt", output_dir_path);

    FILE *file = fopen(metadata_file_path, "w");
    if (file) {
        fprintf(file, "Files written: %d\n", metadata->files_written);
        fprintf(file, "Files read: %d\n", metadata->files_read);
        fprintf(file, "Errors occurred: %d\n", metadata->errors_occurred);
        fprintf(file, "Missing files:\n");

        for (int i = 0; i < metadata->missing_file_count; i++) {
            fprintf(file, " - %s\n", metadata->missing_files[i]);
        }

        fprintf(file, "Processing completed at: %s", ctime(&start_time));
        fclose(file);
    } else {
        printf("Failed to write metadata to %s\n", metadata_file_path);
    }
}


void print_progress_bar(int progress, int total, const char* input_file) {
    int bar_width = (progress * PROGRESS_BAR_WIDTH) / total;
    
    time_t current_time = time(NULL);
    double elapsed_time = difftime(current_time, start_time);
    double estimated_total_time = (elapsed_time / progress) * total;
    double remaining_time = estimated_total_time - elapsed_time;

    int elapsed_min = (int)(elapsed_time / 60);
    int elapsed_sec = (int)(elapsed_time) % 60;
    int remaining_min = (int)(remaining_time / 60);
    int remaining_sec = (int)(remaining_time) % 60;

    printf("\r|");
    for (int i = 0; i < PROGRESS_BAR_WIDTH; i++) {
        if (i < bar_width) printf("\033[32m█\033[0m");  // Green filled
        else printf("\033[91m░\033[0m");  // Red empty
    }
    printf("| %d/%d | Time elapsed: %02d:%02d | Time left: %02d:%02d | Processing: ...%s", 
           progress, total, elapsed_min, elapsed_sec, remaining_min, remaining_sec, remove_last_slash(input_file));
    fflush(stdout);
}

char* remove_last_slash(const char* path){
    if(strncmp(path, input_dir, strlen(input_dir))==0){
        path += strlen(input_dir);
    }
    const char* last_slash = strrchr(path, '/');
    return last_slash ? strndup(path, last_slash - path) : strdup(path);
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

void save_metadata_for_directory(const char *output_dir_path, Metadata *metadata) {
    char metadata_file_path[1028];
    snprintf(metadata_file_path, sizeof(metadata_file_path), "%s/metadata.txt", output_dir_path);

    FILE *file = fopen(metadata_file_path, "w");
    if (file) {
        fprintf(file, "Files written: %d\n", metadata->files_written);
        fprintf(file, "Files read: %d\n", metadata->files_read);
        fprintf(file, "Errors occurred: %d\n", metadata->errors_occurred);
        fprintf(file, "Missing files:\n");

        for (int i = 0; i < metadata->missing_file_count; i++) {
            fprintf(file, " - %s\n", metadata->missing_files[i]);
        }

        fprintf(file, "Processing completed at: %s", ctime(&start_time));
        fclose(file);
    } else {
        printf("Failed to write metadata to %s\n", metadata_file_path);
    }
}

void traverse_directory(const char *dir_path, const char *input_dir) {
    DIR *dir = opendir(dir_path);
    if (!dir) return;

    struct dirent *entry;
    char path[1024];
    Metadata *metadata = &metadata_array[metadata_index++];
    metadata->files_written = 0;
    metadata->files_read = 0;
    metadata->errors_occurred = 0;
    metadata->missing_file_count = 0;

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR) {
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
            snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
            traverse_directory(path, input_dir);
        } else if (strstr(entry->d_name, ".dat")) {
            snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);

            FileTask *task = malloc(sizeof(FileTask));
            task->input_file = strdup(path);
            task->input_dir = input_dir;

            enqueue_task(task);
            metadata->files_read++;
        }
    }
    closedir(dir);

    // Save metadata for the directory after processing its files
    save_metadata_for_directory(dir_path, metadata);
}

void enqueue_task(FileTask *task) {
    pthread_mutex_lock(&queue_mutex);
    while (queue_count >= MAX_QUEUE_SIZE) {
        pthread_cond_wait(&queue_cond, &queue_mutex);
    }
    task_queue[queue_back] = task;
    queue_back = (queue_back + 1) % MAX_QUEUE_SIZE;
    queue_count++;
    pthread_cond_signal(&queue_cond);
    pthread_mutex_unlock(&queue_mutex);
}

FileTask *dequeue_task() {
    pthread_mutex_lock(&queue_mutex);
    while (queue_count == 0) {
        pthread_cond_wait(&queue_cond, &queue_mutex);
    }
    FileTask *task = task_queue[queue_front];
    queue_front = (queue_front + 1) % MAX_QUEUE_SIZE;
    queue_count--;
    pthread_cond_signal(&queue_cond);
    pthread_mutex_unlock(&queue_mutex);
    return task;
}


void *worker_thread(void *arg) {
    while (1) {
        FileTask *task = dequeue_task();
        if (task == NULL) break;

        // Find the corresponding metadata for the directory
        Metadata *metadata = NULL;
        for (int i = 0; i < metadata_index; i++) {
            if (strncmp(task->input_file, metadata_array[i].input_dir, strlen(metadata_array[i].input_dir)) == 0) {
                metadata = &metadata_array[i];
                break;
            }
        }
        if (!metadata) {
            printf("No metadata found for directory %s\n", task->input_dir);
            free((void *)task->input_file);
            free(task);
            continue;
        }

        int result = process_file(task->input_file, task->input_dir);
        if (result == 0) {
            metadata->files_written++;
        } else {
            metadata->errors_occurred++;
            if (metadata->missing_file_count < MAX_MISSING_FILES) {
                metadata->missing_files[metadata->missing_file_count++] = strdup(task->input_file);
            }
        }

        pthread_mutex_lock(&progress_mutex);
        processed_files++;
        if (processed_files % 200 == 0 || processed_files == total_files) {
            print_progress_bar(processed_files, total_files, task->input_file);
        }
        pthread_mutex_unlock(&progress_mutex);

        free((void *)task->input_file);
        free(task);
    }

    return NULL;
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

    // char date_dir[1024];
    // strncpy(date_dir, relative_path, sizeof(date_dir) - 1);
    // date_dir[sizeof(date_dir) - 1] = '\0';
    // char *slash_pos = strchr(date_dir, '/');
    // if (slash_pos) {
    //     *slash_pos = '\0';
    // }

    const char* last_slash = strrchr(relative_path, '/');
    last_slash++;
    const char* date_dir = strndup(last_slash, 8);

    char output_dir_path[1028];
    snprintf(output_dir_path, sizeof(output_dir_path), "%s/%s", output_dir, date_dir);

    // Ensure all intermediate directories in the output path are created
    create_output_directory(output_dir_path);

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

int main() {
    printf("Counting files...\n");
    count_files(input_dir);
    if (total_files == 0) {
        printf("No .dat files found.\n");
        return 0;
    }
    printf("Total files: %d\n", total_files);

    int max_threads = sysconf(_SC_NPROCESSORS_ONLN);
    if (max_threads < 1) max_threads = 1;
    printf("Using %d out of %d threads.\n", max_threads - 2, max_threads);

    start_time = time(NULL); 

    pthread_t threads[max_threads - 2];
    for (int i = 0; i < max_threads - 2; i++) {
        pthread_create(&threads[i], NULL, worker_thread, NULL);
    }

    traverse_directory(input_dir, input_dir);

    for (int i = 0; i < max_threads - 2; i++) {
        enqueue_task(NULL); // Signal termination to worker threads
    }
    for (int i = 0; i < max_threads - 2; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("\nProcessing complete.\n");
    return 0;
}
