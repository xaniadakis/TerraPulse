
#include "feature.h"
#include <stdio.h>
#include <math.h>

// Helper function to calculate the mean of an array
double calculate_mean(double *signal, int length) {
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += signal[i];
    }
    return sum / length;
}

// Helper function to calculate the standard deviation of an array
double calculate_std(double *signal, int length, double mean) {
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += (signal[i] - mean) * (signal[i] - mean);
    }
    return sqrt(sum / length);
}

// Helper function to calculate the root mean square (RMS)
double calculate_rms(double *signal, int length) {
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += signal[i] * signal[i];
    }
    return sqrt(sum / length);
}

// Helper function to calculate the zero-crossing rate (ZCR)
int calculate_zcr(double *signal, int length) {
    int zero_crossings = 0;
    for (int i = 1; i < length; i++) {
        if ((signal[i-1] > 0 && signal[i] < 0) || (signal[i-1] < 0 && signal[i] > 0)) {
            zero_crossings++;
        }
    }
    return zero_crossings;
}

// Helper function to calculate skewness
double calculate_skewness(double *signal, int length, double mean, double std) {
    double skewness = 0.0;
    for (int i = 0; i < length; i++) {
        skewness += pow((signal[i] - mean) / std, 3);
    }
    return skewness / length;
}

// Helper function to calculate kurtosis
double calculate_kurtosis(double *signal, int length, double mean, double std) {
    double kurtosis = 0.0;
    for (int i = 0; i < length; i++) {
        kurtosis += pow((signal[i] - mean) / std, 4);
    }
    return kurtosis / length - 3.0;
}

// Example function to extract time-domain features
TimeDomainFeatures extract_time_domain_features(double *signal, int length) {
    TimeDomainFeatures features;

    // Calculate and store each feature in the struct
    features.mean = calculate_mean(signal, length);
    features.std = calculate_std(signal, length, features.mean);
    features.rms = calculate_rms(signal, length);
    features.zcr = calculate_zcr(signal, length);
    features.skewness = calculate_skewness(signal, length, features.mean, features.std);
    features.kurtosis = calculate_kurtosis(signal, length, features.mean, features.std);

    return features;

}

void extract_frequency_domain_features(double *psd, int length, double sampling_frequency, double min_frequency, double max_frequency) {
    double peak_frequencies[NUM_PEAKS] = {0.0};
    double peak_psd_values[NUM_PEAKS] = {0.0};
    double spectral_centroid = 0.0;
    double spectral_bandwidth = 0.0;
    double total_psd = 0.0;

    for (int i = 0; i < length; i++) {
        double frequency = (double)i * sampling_frequency / length;

        if (frequency >= min_frequency && frequency <= max_frequency) {
            total_psd += psd[i];
            spectral_centroid += frequency * psd[i];

            for (int j = 0; j < NUM_PEAKS; j++) {
                if (psd[i] > peak_psd_values[j]) {
                    for (int k = NUM_PEAKS - 1; k > j; k--) {
                        peak_psd_values[k] = peak_psd_values[k - 1];
                        peak_frequencies[k] = peak_frequencies[k - 1];
                    }
                    peak_psd_values[j] = psd[i];
                    peak_frequencies[j] = frequency;
                    break;
                }
            }
        }
    }

    if (total_psd > 0) {
        spectral_centroid /= total_psd;
    }

    for (int i = 0; i < length; i++) {
        double frequency = (double)i * sampling_frequency / length;

        if (frequency >= min_frequency && frequency <= max_frequency) {
            spectral_bandwidth += pow(frequency - spectral_centroid, 2) * psd[i];
        }
    }

    if (total_psd > 0) {
        spectral_bandwidth = sqrt(spectral_bandwidth / total_psd);
    }

    printf("Top 7 Peak Frequencies and PSD Values:\n");
    for (int i = 0; i < NUM_PEAKS; i++) {
        printf("Peak %d: Frequency = %f, PSD = %f\n", i + 1, peak_frequencies[i], peak_psd_values[i]);
    }

    printf("Sampling Frequency: %f\n", sampling_frequency);
    printf("Spectral Centroid: %f\n", spectral_centroid);
    printf("Spectral Bandwidth: %f\n", spectral_bandwidth);
}

void extract_frequency_domain_features2(double *psd, int length, double sampling_frequency, double min_frequency, double max_frequency) {
    double max_psd_value = 0.0;
    double max_frequency_value = 0.0;

    double spectral_centroid = 0.0;
    double spectral_bandwidth = 0.0;
    double weighted_sum = 0.0;
    double total_psd_sum = 0.0;
    double weighted_squared_diff_sum = 0.0;

    double frequency_resolution = (sampling_frequency / 2.0) / length;

    int min_index = (int)(min_frequency / frequency_resolution);
    int max_index = (int)(max_frequency / frequency_resolution);

    for (int i = min_index; i <= max_index; i++) {
        double frequency = (double)i * frequency_resolution;

        if (psd[i] > max_psd_value) {
            max_psd_value = psd[i];
            max_frequency_value = frequency;
        }

        weighted_sum += psd[i] * frequency;
        total_psd_sum += psd[i];
    }

    if (total_psd_sum > 0) {
        spectral_centroid = weighted_sum / total_psd_sum;
    }

    for (int i = min_index; i <= max_index; i++) {
        double frequency = (double)i * frequency_resolution;
        weighted_squared_diff_sum += psd[i] * pow(frequency - spectral_centroid, 2);
    }

    if (total_psd_sum > 0) {
        spectral_bandwidth = sqrt(weighted_squared_diff_sum / total_psd_sum);
    }

    printf("Maximum PSD value: %.6f at frequency: %.6f Hz\n", max_psd_value, max_frequency_value);
    printf("Spectral Centroid: %.6f Hz\n", spectral_centroid);
    printf("Spectral Bandwidth: %.6f Hz\n", spectral_bandwidth);
}
