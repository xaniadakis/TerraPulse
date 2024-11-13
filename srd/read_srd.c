#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <zstd.h>
#include <fftw3.h>

#define DATALOGGERID 0xCAD0FFE51513FFDC
#define MAX_VAL_OLD 65535.0
#define MAX_VAL_NEW 32767.0

typedef struct {
    struct timespec timestamp;
    double sampling_frequency;
    int channel;
    double battery_voltage;
    int success;
} SrdInfo;

SrdInfo get_srd_info(const char *filename) {
    SrdInfo info = {{0, 0}, -1, 0, 0.0, 0};
    FILE *file = fopen(filename, "rb");
    if (!file) return info;

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    if (file_size < 1024) {
        fclose(file);
        return info;
    }

    fseek(file, 0, SEEK_SET);
    uint64_t logger_id;
    fread(&logger_id, sizeof(uint64_t), 1, file);
    if (logger_id != DATALOGGERID) {
        fclose(file);
        return info;
    }

    // Read timestamp components directly
    uint8_t sec, min, hour, day, month, year;
    fread(&sec, sizeof(uint8_t), 1, file);
    fread(&min, sizeof(uint8_t), 1, file);
    fread(&hour, sizeof(uint8_t), 1, file);
    fread(&day, sizeof(uint8_t), 1, file);
    fread(&month, sizeof(uint8_t), 1, file);
    fread(&year, sizeof(uint8_t), 1, file);
    year += 1970;  // Adjust year as in Python

    // Construct tm structure for date and time
    struct tm date_tm = {0};
    date_tm.tm_sec = sec;
    date_tm.tm_min = min;
    date_tm.tm_hour = hour;
    date_tm.tm_mday = day;
    date_tm.tm_mon = month - 1;
    date_tm.tm_year = year - 1900;

    // Set the timestamp
    time_t date_seconds = mktime(&date_tm);
    struct timespec date = {0};
    date.tv_sec = date_seconds;
    date.tv_nsec = 400000000;  // Apply 400 ms offset as per your original data

    // Correction based on date ranges, similar to Python's adjustment
    struct tm t0_tm = {.tm_year = 116, .tm_mon = 0, .tm_mday = 1};  // 2016-01-01
    struct tm t1_tm = {.tm_year = 117, .tm_mon = 7, .tm_mday = 1};  // 2017-08-01
    time_t t0 = mktime(&t0_tm);
    time_t t1 = mktime(&t1_tm);

    if (t0 < date.tv_sec && date.tv_sec < t1) {
        double tslop = 480.0 / 600.0;  // seconds offset per day
        int days_diff = (date.tv_sec - t0) / (24 * 3600);
        date.tv_sec -= (int)(days_diff * tslop);
    }
    info.timestamp = date;

    // Read sampling frequency, channel, and battery voltage
    fseek(file, 15, SEEK_SET);
    float fs;
    fread(&fs, sizeof(float), 1, file);
    info.sampling_frequency = fs;
    fread(&info.channel, sizeof(uint8_t), 1, file);
    float vbat;
    fread(&vbat, sizeof(float), 1, file);
    info.battery_voltage = vbat;

    info.success = 1;
    fclose(file);
    return info;
}

void read_srd_file(const char *filename, SrdInfo *info, double **x, double **y, int *length) {
    *x = NULL;
    *y = NULL;
    *length = 0;

    FILE *file = fopen(filename, "rb");
    if (!file || !info->success || info->sampling_frequency <= 0) return;

    fseek(file, 528, SEEK_END);
    long file_size = ftell(file) - 528;
    fseek(file, 528, SEEK_SET);

    int num_samples = file_size / sizeof(uint16_t);
    *length = num_samples;

    uint16_t *raw_data = malloc(num_samples * sizeof(uint16_t));
    if (!raw_data) {
        fclose(file);
        return;
    }

    fread(raw_data, sizeof(uint16_t), num_samples, file);
    fclose(file);

    *x = malloc(num_samples * sizeof(double));
    if (!*x) {
        free(raw_data);
        return;
    }

    double max_val = (info->timestamp.tv_sec < mktime(&(struct tm){.tm_year = 117, .tm_mon = 7, .tm_mday = 10})) ? MAX_VAL_OLD : MAX_VAL_NEW;
    if (info->channel == 0) {
        for (int i = 0; i < num_samples; i++) {
            (*x)[i] = raw_data[i] * 4.096 / max_val - 2.048;
        }
    } else {
        *y = malloc(num_samples / 2 * sizeof(double));
        if (!*y) {
            free(raw_data);
            free(*x);
            *x = NULL;
            return;
        }
        for (int i = 0; i < num_samples / 2; i++) {
            (*x)[i] = raw_data[2 * i] * 4.096 / max_val - 2.048;
            (*y)[i] = raw_data[2 * i + 1] * 4.096 / max_val - 2.048;
        }
    }
    free(raw_data);
}

void print_srd_info(SrdInfo *info, double *x, int length) {
    struct tm *local_time = localtime(&info->timestamp.tv_sec);
    char time_buf[100];
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", local_time);
    printf("Timestamp: %s.%03ld\n", time_buf, info->timestamp.tv_nsec / 1000000);
    printf("Sampling Frequency (fs): %.1f Hz\n", info->sampling_frequency);
    printf("Battery Voltage (fs): %.4f V\n", info->battery_voltage);

    printf("Channel Count: %s\n", info->channel ? "Dual (x and y)" : "Single (x)");
    printf("First 10 Samples (Channel x): ");
    for (int i = 0; i < 10 && i < length; i++) {
        printf("%.8f ", x[i]);
    }
    printf("\nLast 10 Samples (Channel x): ");
    for (int i = length-11; i < length; i++) {
        printf("%.8f ", x[i]);
    }
    printf("\n");
    printf("Number of Samples in x: %d\n", length);
    printf("%.2f seconds saved in file\n", length / info->sampling_frequency);
}

int main() {
    const char *filename = "/mnt/e/KalpakiSortedData/160202_1/0010.SRD";
    SrdInfo info = get_srd_info(filename);
    if (!info.success) {
        printf("Error reading SRD file\n");
        return 1;
    }

    double *x = NULL, *y = NULL;
    int length;
    read_srd_file(filename, &info, &x, &y, &length);

    if (!x) {
        printf("Failed to load data from SRD file\n");
        return 1;
    }

    print_srd_info(&info, x, length);

    free(x);
    free(y);

    return 0;
}
