//
// Created by vag on 10/15/24.
//

#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#define NUM_PEAKS 7

// Define a struct to hold the time-domain features
typedef struct {
    double mean;
    double std;
    double rms;
    int zcr;
    double skewness;
    double kurtosis;
} TimeDomainFeatures;

double calculate_mean(double *signal, int length);
double calculate_std(double *signal, int length, double mean);
double calculate_rms(double *signal, int length);
int calculate_zcr(double *signal, int length);
double calculate_skewness(double *signal, int length, double mean, double std);
double calculate_kurtosis(double *signal, int length, double mean, double std);

TimeDomainFeatures extract_time_domain_features(double *signal, int length);
void extract_frequency_domain_features(double *psd, int length, double sampling_frequency, double min_frequency, double max_frequency);
void extract_frequency_domain_features2(double *psd, int length, double sampling_frequency, double min_frequency, double max_frequency);

#endif //FEATURE_EXTRACTION_H
