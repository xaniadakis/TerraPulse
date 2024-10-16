
#include "signanalysis.h"
#include "feature.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>

double known_schumann_frequencies[] = {7.83, 14.3, 20.8, 27.3, 33.8, 39.9, 46.6};

void compute_psd(double *signal, int length, char *output_file, double sampling_frequency, double min_frequency, double max_frequency) {
    fftw_complex *out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (length / 2 + 1));
    double *in = (double *) fftw_malloc(sizeof(double) * length);
    double *psd = (double *) malloc(sizeof(double) * (length / 2 + 1));

    // Copy the input signal into the FFTW input buffer
    for (int i = 0; i < length; i++) {
        in[i] = signal[i];
    }

    // Create and execute the FFT plan
    fftw_plan plan = fftw_plan_dft_r2c_1d(length, in, out, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Compute the Power Spectral Density (PSD)
    for (int i = 0; i < length / 2 + 1; i++) {
        psd[i] = (out[i][0] * out[i][0] + out[i][1] * out[i][1]) / (length * sampling_frequency);
    }

    // Open the output file for writing
    FILE *f = fopen(output_file, "w");
    if (!f) {
        perror("File opening failed");
        return;
    }

    // Write frequency and PSD to the output file
    for (int i = 0; i < length / 2 + 1; i++) {
        double frequency = (double) i * sampling_frequency / length;
        if (frequency >= min_frequency && frequency <= max_frequency) {
            fprintf(f, "%.6f\t%.6f\n", frequency, psd[i]);
        }
    }

    // Close the output file
    fclose(f);

    // Call a function to extract frequency domain features (if needed)
    extract_frequency_domain_features2(psd, length / 2 + 1, sampling_frequency, min_frequency, max_frequency);

    // Cleanup FFT resources
    fftw_destroy_plan(plan);
    fftw_free(out);
    fftw_free(in);
    free(psd);
}

Harmonic* analyze_schumann_harmonics(double *signal, int length, double sampling_frequency) {
    fftw_complex *out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (length / 2 + 1));
    double *in = (double *) fftw_malloc(sizeof(double) * length);
    double *psd = (double *) malloc(sizeof(double) * (length / 2 + 1));
    Harmonic *harmonics = (Harmonic *) malloc(sizeof(Harmonic) * NUM_HARMONICS);

    if (!out || !in || !psd || !harmonics) {
        fprintf(stderr, "Memory allocation failed!\n");
        fftw_free(out);
        fftw_free(in);
        free(psd);
        free(harmonics);
        return NULL;
    }

    for (int i = 0; i < length; i++) {
        in[i] = signal[i];
    }

    fftw_plan plan = fftw_plan_dft_r2c_1d(length, in, out, FFTW_ESTIMATE);
    fftw_execute(plan);

    for (int i = 0; i < length / 2 + 1; i++) {
        psd[i] = (out[i][0] * out[i][0] + out[i][1] * out[i][1]) / length;
    }

    for (int h = 0; h < NUM_HARMONICS; h++) {
        double target_frequency = known_schumann_frequencies[h];
        double search_range = 2.5;
        double max_psd_value = 0.0;
        double peak_frequency = 0.0;

        for (int i = 1; i < length / 2 + 1; i++) {
            double frequency = (double) i * sampling_frequency / length;
            if (fabs(frequency - target_frequency) <= search_range && psd[i] > max_psd_value) {
                max_psd_value = psd[i];
                peak_frequency = frequency;
            }
        }
        harmonics[h].target_frequency = target_frequency;
        harmonics[h].peak_frequency = peak_frequency;
        harmonics[h].amplitude = max_psd_value;
    }

    fftw_destroy_plan(plan);
    fftw_free(out);
    fftw_free(in);
    free(psd);
    return harmonics;

}

void find_modes(double *signal, int length, double sampling_frequency, double min_frequency, double max_frequency, int num_modes) {
    fftw_complex *out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (length / 2 + 1));
    double *in = (double *) fftw_malloc(sizeof(double) * length);
    double *psd = (double *) malloc(sizeof(double) * (length / 2 + 1));

    if (!out || !in || !psd) {
        fprintf(stderr, "Memory allocation failed!\n");
        fftw_free(out);
        fftw_free(in);
        free(psd);
        return;
    }

    for (int i = 0; i < length; i++) {
        in[i] = signal[i];
    }

    fftw_plan plan = fftw_plan_dft_r2c_1d(length, in, out, FFTW_ESTIMATE);
    fftw_execute(plan);

    for (int i = 0; i < length / 2 + 1; i++) {
        psd[i] = (out[i][0] * out[i][0] + out[i][1] * out[i][1]) / length;
    }

    double *modes_frequency = (double *) malloc(num_modes * sizeof(double));
    double *modes_amplitude = (double *) malloc(num_modes * sizeof(double));

    if (!modes_frequency || !modes_amplitude) {
        fprintf(stderr, "Memory allocation failed!\n");
        fftw_destroy_plan(plan);
        fftw_free(out);
        fftw_free(in);
        free(psd);
        return;
    }

    for (int n = 0; n < num_modes; n++) {
        double max_psd_value = 0.0;
        int peak_index = 0;

        for (int i = 1; i < length / 2 + 1; i++) {
            double frequency = (double) i * sampling_frequency / length;
            if (frequency >= min_frequency && frequency <= max_frequency && psd[i] > max_psd_value) {
                max_psd_value = psd[i];
                peak_index = i;
            }
        }

        modes_frequency[n] = (double) peak_index * sampling_frequency / length;
        modes_amplitude[n] = max_psd_value;

        int suppress_range = 1000;
        int start = (peak_index - suppress_range >= 0) ? peak_index - suppress_range : 0;
        int end = (peak_index + suppress_range < length / 2 + 1) ? peak_index + suppress_range : length / 2;
        for (int i = start; i <= end; i++) {
            psd[i] = 0.0;
        }
    }

    printf("Top %d modes:\n", num_modes);
    printf("Mode\tF (Hz)\t\tAmplitude\n");
    for (int i = 0; i < num_modes; i++) {
        printf("%d\t%.6f\t%.6f\n", i + 1, modes_frequency[i], modes_amplitude[i]);
    }

    fftw_destroy_plan(plan);
    fftw_free(out);
    fftw_free(in);
    free(psd);
    free(modes_frequency);
    free(modes_amplitude);
}

void downsample_signal(double *input_signal, double *downsampled_signal, int downsampled_length) {
    for (int i = 0; i < downsampled_length; i++) {
        double avg_value = 0.0;

        for (int j = 0; j < DOWNSAMPLING_FACTOR; j++) {
            avg_value += input_signal[i * DOWNSAMPLING_FACTOR + j];
        }
        avg_value /= DOWNSAMPLING_FACTOR;
        downsampled_signal[i] = avg_value;
    }
}
