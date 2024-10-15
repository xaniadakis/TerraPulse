#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <math.h>
#include <fftw3.h>
#include "feature.h"
#include "io.h"
#include "signanalysis.h"
#include <stdbool.h>

int main() {
    // Measure the total start time
    clock_t total_start = clock();

    const char *input_dir = "/media/vag/Users/echan/Documents/Parnon/20230106/";
    const char *output_dir = "./output/";
    double sampling_frequency = 5e6 / 128 / 13;
    double downsampled_frequency = 5e6 / 128 / 13 / DOWNSAMPLING_FACTOR;
    double min_freq = 3.0;
    double max_freq = 48.0;

    // Create output directory if it doesn't exist
    struct stat st = {0};
    if (stat(output_dir, &st) == -1) {
        mkdir(output_dir, 0700);
    }

    DIR *dir = opendir(input_dir);
    struct dirent *entry;

    if (dir == NULL) {
        perror("Failed to open directory");
        return EXIT_FAILURE;
    }

    while ((entry = readdir(dir)) != NULL) {
        // Process only .dat files
        if (strstr(entry->d_name, ".dat")) {
            char input_dat_file[1024];
            snprintf(input_dat_file, sizeof(input_dat_file), "%s%s", input_dir, entry->d_name);

            // Generate output file name
            char output_file[1024];
            snprintf(output_file, sizeof(output_file), "%s%s.txt", output_dir, strtok(entry->d_name, "."));

            int *HNS = NULL, *HEW = NULL;
            int nr = 0;

            // Start reading the DAT file
            clock_t start_reading = clock();
            read_dat_file(input_dat_file, &HNS, &HEW, &nr);
            clock_t end_reading = clock();
            double reading_time = (double) (end_reading - start_reading) / CLOCKS_PER_SEC;

            // Start calibrating
            clock_t start_calibrating = clock();
            double *calibrated_HNS = NULL, *calibrated_HEW = NULL;
            calibrate_HYL(HNS, HEW, nr, &calibrated_HNS, &calibrated_HEW);
            clock_t end_calibrating = clock();
            double calibration_time = (double) (end_calibrating - start_calibrating) / CLOCKS_PER_SEC;

            printf("\n%.4s-%.2s-%.2s %.2s:%.2s\n", entry->d_name, entry->d_name + 4,
                   entry->d_name + 6, entry->d_name + 8, entry->d_name + 10);

            printf("HNS:\n");
            extract_time_domain_features(calibrated_HNS, nr);

            printf("HEW:\n");
            extract_time_domain_features(calibrated_HEW, nr);

            // Call the PSD computation function
            //             char psd_output_file[1024];
            //             snprintf(psd_output_file, sizeof(psd_output_file), "%s%s_psd.txt", output_dir, strtok(entry->d_name, "."));
            //            find_modes(downsampled_HNS, downsampled_length , psd_output_file, sampling_frequency, min_freq, max_freq, 7, 1); // Call with HNS signal
            //            find_modes(downsampled_HNS, downsampled_length, sampling_frequency, min_freq, max_freq, 7); // Call with HNS signal
            analyze_schumann_harmonics(calibrated_HNS, nr, sampling_frequency);

            int downsampled_length = nr / DOWNSAMPLING_FACTOR;
            double *downsampled_HNS = (double *) malloc(downsampled_length * sizeof(double));
            double *downsampled_HEW = (double *) malloc(downsampled_length * sizeof(double));

            // Downsample the signals
            downsample_signal(calibrated_HNS, downsampled_HNS, downsampled_length);
            downsample_signal(calibrated_HEW, downsampled_HEW, downsampled_length);

            // Save downsampled signals to file
            save_signals(downsampled_HNS, downsampled_HEW, downsampled_length, output_file);


            // Print timing information
            // FILE *file = fopen(input_dat_file, "rb");
            // fseek(file, 0, SEEK_END);
            // long file_size = ftell(file);
            // fclose(file);

            // double file_size_mb = (double)file_size / (1024 * 1024);
            // double initial_frequency = 5e6 / 128 / 13;  // Example frequency value
            // printf("\nProcessed file: %s", input_dat_file);
            // printf("\nDAT file size: %.2f MB, Samples: %d, Initial Frequency: %.2f Hz", file_size_mb, nr, initial_frequency);
            // printf("\nTime to read: %.4f seconds", reading_time);
            // printf("\nTime to calibrate: %.4f seconds", calibration_time);
            //            printf("%s\n", output_file);

            // Free allocated memory
            free(HNS);
            free(HEW);
            free(calibrated_HNS);
            free(calibrated_HEW);
            free(downsampled_HNS);
            free(downsampled_HEW);
        }
    }

    closedir(dir);

    // Measure total end time
    clock_t total_end = clock();
    double total_time = (double) (total_end - total_start) / CLOCKS_PER_SEC;
    printf("\nTotal time for the whole process: %.4f seconds", total_time);

    return 0;
}
