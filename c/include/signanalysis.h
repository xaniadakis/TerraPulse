//
// Created by vag on 10/15/24.
//

#ifndef SIGNAL_ANALYSIS_H
#define SIGNAL_ANALYSIS_H

#define DOWNSAMPLING_FACTOR 30
#define NUM_HARMONICS 7

extern double known_schumann_frequencies[];

void compute_psd(double *signal, int length, char *output_file, double sampling_frequency, double min_frequency, double max_frequency);
void analyze_schumann_harmonics(double *signal, int length, double sampling_frequency);
void find_modes(double *signal, int length, double sampling_frequency, double min_frequency, double max_frequency, int num_modes);
void downsample_signal(double *input_signal, double *downsampled_signal, int downsampled_length);

#endif // SIGNAL_ANALYSIS_H
