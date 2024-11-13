//
// Created by vag on 10/15/24.
//

#ifndef SIGNAL_ANALYSIS_H
#define SIGNAL_ANALYSIS_H

#include "srd.h"
#define DOWNSAMPLING_FACTOR 30
#define NUM_HARMONICS 7

typedef struct {
    double target_frequency;
    double peak_frequency;
    double amplitude;
} Harmonic;

extern double known_schumann_frequencies[];

void compute_psd(double *signal, int length, double *frequencies, double *psd, char *output_file, double sampling_frequency, double min_frequency, double max_frequency);
Harmonic* analyze_schumann_harmonics(double *signal, int length, double sampling_frequency);
void find_modes(double *signal, int length, double sampling_frequency, double min_frequency, double max_frequency, int num_modes);
void downsample_dat_signal(double *input_signal, double *downsampled_signal, int downsampled_length);

// void downsample_srd_signal(const double *input, int input_size, int downsample_factor, const char *filename);
void downsample_srd_signal(SrdData data, int downsample_factor, const char *filename);

#endif // SIGNAL_ANALYSIS_H
