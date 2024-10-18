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
#include "lorentzianLib.h"  // Include the generated header file

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
	printf("| %.2f%%", (progress / (double)total) * 100); // Display percentage
	fflush(stdout); // Ensure immediate output
}

int main() {
    // Measure the total start time
    clock_t total_start = clock();

    const char *input_dir = "/media/vag/Users/echan/Documents/Parnon/20230106/";
    const char *output_dir = "./output/";
//    double sampling_frequency = 5e6 / 128 / 13;
    // double downsampled_frequency = 5e6 / 128 / 13 / DOWNSAMPLING_FACTOR;
//    double min_freq = 3.0;
//    double max_freq = 48.0;

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

	int total_files = 0;
	while ((entry = readdir(dir)) != NULL) {
		if (strstr(entry->d_name, ".dat")) {
			total_files++;
		}
	}
	rewinddir(dir); // Reset directory pointer to the beginning

	int processed_files = 0;
    while ((entry = readdir(dir)) != NULL) {
        // Process only .dat files
        if (strstr(entry->d_name, ".dat")) {
            char input_dat_file[1024];
            snprintf(input_dat_file, sizeof(input_dat_file), "%s%s", input_dir, entry->d_name);

            // Generate output file name
            char output_file[1024];
            snprintf(output_file, sizeof(output_file), "%s%s.txt", output_dir, strtok(entry->d_name, "."));

            int *HNS = NULL, *HEW = NULL, nr = 0;
            read_dat_file(input_dat_file, &HNS, &HEW, &nr);

            double *calibrated_HNS = NULL, *calibrated_HEW = NULL;
            calibrate_HYL(HNS, HEW, nr, &calibrated_HNS, &calibrated_HEW);

			TimeDomainFeatures* hns_features = NULL;
//			extract_time_domain_features(calibrated_HNS, nr);
        	TimeDomainFeatures* hew_features = NULL;
//        	extract_time_domain_features(calibrated_HEW, nr);

            // Call the PSD computation function
//			char psd_output_file[1024];
//			snprintf(psd_output_file, sizeof(psd_output_file), "%s%s_psd.txt", output_dir, strtok(entry->d_name, "."));
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

            // Downsample the signals
            downsample_signal(calibrated_HNS, downsampled_HNS, downsampled_length);
            downsample_signal(calibrated_HEW, downsampled_HEW, downsampled_length);

            // Save downsampled signals to file
            save_signals(downsampled_HNS, downsampled_HEW, hns_features, hew_features, NULL, downsampled_length, output_file);

            // Free allocated memory
            free(HNS);
            free(HEW);
            free(calibrated_HNS);
            free(calibrated_HEW);
            free(downsampled_HNS);
            free(downsampled_HEW);
            // free(harmonics);
        }
    	processed_files++;
    	print_progress_bar(processed_files, total_files);

        if (TEST == 1 && processed_files > 20){
            break;
        }
    }

    closedir(dir);

    // Measure total end time
    clock_t total_end = clock();
    double total_time = (double) (total_end - total_start) / CLOCKS_PER_SEC;
    printf("\nTotal time for the whole process: %.4f seconds\n", total_time);

    return 0;
}

