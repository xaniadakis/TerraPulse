#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <math.h>
#include <fftw3.h> // Ensure you have FFTW installed for FFT computations

void compute_psd(double *signal, int length, char *output_file, double sampling_frequency) {
    // Allocate memory for FFTW input and output
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (length / 2 + 1));
    double *in = (double*) fftw_malloc(sizeof(double) * length);
    double *psd = (double*) malloc(sizeof(double) * (length / 2 + 1));

    // Copy the signal to FFTW input array
    for (int i = 0; i < length; i++) {
        in[i] = signal[i];
    }

    // Create a plan for FFT
    fftw_plan plan = fftw_plan_dft_r2c_1d(length, in, out, FFTW_ESTIMATE);

    // Execute the FFT
    fftw_execute(plan);

    // Compute the PSD
    for (int i = 0; i < length / 2 + 1; i++) {
        // PSD = |X(f)|^2 / N
        psd[i] = (out[i][0] * out[i][0] + out[i][1] * out[i][1]) / length;
    }

    // Save PSD to a file for Gnuplot
    FILE *f = fopen(output_file, "w");
    if (!f) {
        perror("File opening failed");
        return;
    }

    for (int i = 0; i < length / 2 + 1; i++) {
        double frequency = (double)i * sampling_frequency / length; // Frequency corresponding to bin
        fprintf(f, "%.6f\t%.6f\n", frequency, psd[i]); // Frequency and PSD value
    }

    fclose(f);

    // Cleanup
    fftw_destroy_plan(plan);
    fftw_free(out);
    fftw_free(in);
    free(psd);
}

void read_dat_file(const char *fn, int **Bx, int **By, int *nr_out) {
    FILE *f = fopen(fn, "rb");
    if (!f) {
        perror("File opening failed");
        return;
    }

    // Skip header (64 bytes)
    fseek(f, 64, SEEK_SET);

    // Determine file size and read data
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f) - 64;
    fseek(f, 64, SEEK_SET);

    uint8_t *data = (uint8_t *)malloc(file_size);
    if (!data) {
        perror("Memory allocation failed");
        fclose(f);
        return;
    }

    fread(data, 1, file_size, f);
    fclose(f);

    int ld = file_size;
    int Bx_size = ld / 4;
    int By_size = ld / 4;

    *Bx = (int *)calloc(Bx_size, sizeof(int));
    *By = (int *)calloc(By_size, sizeof(int));
    if (!(*Bx) || !(*By)) {
        perror("Memory allocation failed");
        free(data);
        return;
    }

    int a = 65536;
    int nr = 0;
    int i = 1;

    // First loop for j in range(89)
    for (int j = 0; j < 89; j++) {
        (*Bx)[nr] = a * ((data[i] & 12) / 4) + data[i + 1] * 256 + data[i + 2];
        (*By)[nr] = a * (data[i] & 3) + data[i + 3] * 256 + data[i + 4];
        nr++;
        i += 5;
    }
    i += 2;

    // Second loop for n in range(8836)
    for (int n = 0; n < 8836; n++) {
        for (int j = 0; j < 102; j++) {
            (*Bx)[nr] = a * ((data[i] & 12) / 4) + data[i + 1] * 256 + data[i + 2];
            (*By)[nr] = a * (data[i] & 3) + data[i + 3] * 256 + data[i + 4];
            nr++;
            i += 5;
        }
        i += 2;
    }

    // Third loop for j in range(82)
    for (int j = 0; j < 82; j++) {
        (*Bx)[nr] = a * ((data[i] & 12) / 4) + data[i + 1] * 256 + data[i + 2];
        (*By)[nr] = a * (data[i] & 3) + data[i + 3] * 256 + data[i + 4];
        nr++;
        i += 5;
    }

    // While loop to trim zeros
    while ((*Bx)[nr] == 0 || (*By)[nr] == 0) {
        nr--;
    }

    // Adjust Bx and By arrays
    int midADC = 1 << 17;  // 2^18 / 2
    for (int k = 0; k <= nr; k++) {
        (*Bx)[k] -= midADC;
        (*By)[k] -= midADC;
    }

    *nr_out = nr;
    free(data);
}

void calibrate_HYL(int *Bx, int *By, int length, double **calibrated_Bx, double **calibrated_By) {
    // Constants
    double a1_mVnT = 55.0;  // [mV/nT] conversion coefficient
    double a2_mVnT = 55.0;  // [mV/nT]

    double a1 = a1_mVnT * 1e-3 / 1e3;  // [V/pT]
    double a2 = a2_mVnT * 1e-3 / 1e3;  // [V/pT]
    double ku = 4.26;  // Amplification in the receiver
    double c1 = a1 * ku;  // System sensitivity
    double c2 = a2 * ku;  // System sensitivity
    double d = (double)(1 << 18);  // 18-bit digital-to-analog converter
    double V = 4.096 * 2;  // [V] Voltage range of digital-to-analog converter

    // Scales
    double scale1 = c1 * d / V;
    double scale2 = c2 * d / V;

    // Allocate memory for calibrated results
    *calibrated_Bx = (double *)malloc(length * sizeof(double));
    *calibrated_By = (double *)malloc(length * sizeof(double));

    if (*calibrated_Bx == NULL || *calibrated_By == NULL) {
        perror("Memory allocation failed");
        return;
    }

    // Calibrate each Bx and By value
    for (int i = 0; i < length; i++) {
        (*calibrated_Bx)[i] = -Bx[i] / scale1;
        (*calibrated_By)[i] = -By[i] / scale2;
    }
}

// Downsample the signal by averaging every 30 samples
void downsample_and_save(double *calibrated_HNS, double *calibrated_HEW, int length, const char *output_file, int factor) {
    FILE *f = fopen(output_file, "w");
    if (!f) {
        perror("File opening failed");
        return;
    }

    int downsampled_length = length / factor;
    for (int i = 0; i < downsampled_length; i++) {
        double avg_HNS = 0.0, avg_HEW = 0.0;

        // Average over the factor (30 samples)
        for (int j = 0; j < factor; j++) {
            avg_HNS += calibrated_HNS[i * factor + j];
            avg_HEW += calibrated_HEW[i * factor + j];
        }
        avg_HNS /= factor;
        avg_HEW /= factor;

        // Write to the file in the format HNS\tHEW\n
        // fprintf(f, "%.6f\t%.6f\n", avg_HNS, avg_HEW);
        fprintf(f, "%d\t%d\n", (int)avg_HNS, (int)avg_HEW);
    }

    fclose(f);
}

int main() {
    // Measure the total start time
    clock_t total_start = clock();

    const char *input_dir = "/media/vag/Users/echan/Documents/Parnon/20230106/";
    const char *output_dir = "./output/";
    int downsampling_factor = 30; // Downsampling by 30

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
            double reading_time = (double)(end_reading - start_reading) / CLOCKS_PER_SEC;

            // Start calibrating
            clock_t start_calibrating = clock();
            double *calibrated_HNS = NULL, *calibrated_HEW = NULL;
            calibrate_HYL(HNS, HEW, nr, &calibrated_HNS, &calibrated_HEW);
            clock_t end_calibrating = clock();
            double calibration_time = (double)(end_calibrating - start_calibrating) / CLOCKS_PER_SEC;

            // Downsample and save the result
            downsample_and_save(calibrated_HNS, calibrated_HEW, nr, output_file, downsampling_factor);

            // // Call the PSD computation function
            // char psd_output_file[1024];
            // snprintf(psd_output_file, sizeof(psd_output_file), "%s%s_psd.txt", output_dir, strtok(entry->d_name, "."));
            // double sampling_frequency = 5e6 / 128 / 13 / downsampling_factor; // 100 Hz as given
            // compute_psd(calibrated_HNS, nr / downsampling_factor, psd_output_file, sampling_frequency); // Call with HNS signal

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
        }
    }

    closedir(dir);

    // Measure total end time
    clock_t total_end = clock();
    double total_time = (double)(total_end - total_start) / CLOCKS_PER_SEC;
//    printf("\nTotal time for the whole process: %.4f seconds", total_time);

    return 0;
}