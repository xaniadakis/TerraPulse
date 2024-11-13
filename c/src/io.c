
#include "io.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


int read_dat_file(const char *fn, int **Bx, int **By, int *nr_out) {
    FILE *f = fopen(fn, "rb");
    if (!f) {
        perror("read_dat_file: File opening failed");
        return 1;
    }

    fseek(f, 64, SEEK_SET);  // Skip header
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f) - 64;
    fseek(f, 64, SEEK_SET);

    uint8_t *data = (uint8_t *) malloc(file_size);
    if (!data) {
        perror("Memory allocation failed");
        fclose(f);
        return 2;
    }

    fread(data, 1, file_size, f);
    fclose(f);

    int ld = file_size;
    int Bx_size = ld / 4;
    int By_size = ld / 4;

    *Bx = (int *) calloc(Bx_size, sizeof(int));
    *By = (int *) calloc(By_size, sizeof(int));
    if (!(*Bx) || !(*By)) {
        perror("Memory allocation failed");
        free(data);
        return 2;
    }

    int a = 65536;
    int nr = 0;
    int i = 1;

    // Adjusted bounds check with more verbose error reporting
    for (int j = 0; j < 89; j++) {
        if (i + 4 >= file_size) {
            fprintf(stderr, "Out of bounds in read_dat_file (loop 1): %d/%ld, file: %s\n", i + 4, file_size, fn);
            free(data);
            free(*Bx);
            free(*By);
            return -1;
        }
        (*Bx)[nr] = a * ((data[i] & 12) / 4) + data[i + 1] * 256 + data[i + 2];
        (*By)[nr] = a * (data[i] & 3) + data[i + 3] * 256 + data[i + 4];
        nr++;
        i += 5;
    }
    i += 2;

    // Loop 2 with error reporting
    for (int n = 0; n < 8836; n++) {
        for (int j = 0; j < 102; j++) {
            if (i + 4 >= file_size) {
                fprintf(stderr, "Out of bounds in read_dat_file (loop 2): %d/%ld, file: %s\n", i + 4, file_size, fn);
                free(data);
                free(*Bx);
                free(*By);
                return -1;
            }
            (*Bx)[nr] = a * ((data[i] & 12) / 4) + data[i + 1] * 256 + data[i + 2];
            (*By)[nr] = a * (data[i] & 3) + data[i + 3] * 256 + data[i + 4];
            nr++;
            i += 5;
        }
        i += 2;
    }

    // Loop 3 with error reporting
    for (int j = 0; j < 82; j++) {
        if (i + 4 >= file_size) {
            fprintf(stderr, "Out of bounds in read_dat_file (loop 3): %d/%ld, file: %s\n", i + 4, file_size, fn);
            free(data);
            free(*Bx);
            free(*By);
            return -1;
        }
        (*Bx)[nr] = a * ((data[i] & 12) / 4) + data[i + 1] * 256 + data[i + 2];
        (*By)[nr] = a * (data[i] & 3) + data[i + 3] * 256 + data[i + 4];
        nr++;
        i += 5;
    }

    // Adjust for trailing zero values if necessary
    while (nr > 0 && ((*Bx)[nr] == 0 || (*By)[nr] == 0)) {
        nr--;
    }

    int midADC = 1 << 17;
    for (int k = 0; k <= nr; k++) {
        (*Bx)[k] -= midADC;
        (*By)[k] -= midADC;
    }

    *nr_out = nr;
    free(data);
    return 0;
}

void calibrate_HYL(int *Bx, int *By, int length, double **calibrated_Bx, double **calibrated_By) {
    double a1_mVnT = 55.0;
    double a2_mVnT = 55.0;

    double a1 = a1_mVnT * 1e-3 / 1e3;
    double a2 = a2_mVnT * 1e-3 / 1e3;
    double ku = 4.26;
    double c1 = a1 * ku;
    double c2 = a2 * ku;
    double d = (double)(1 << 18);
    double V = 4.096 * 2;

    double scale1 = c1 * d / V;
    double scale2 = c2 * d / V;

    *calibrated_Bx = (double *) malloc(length * sizeof(double));
    *calibrated_By = (double *) malloc(length * sizeof(double));

    if (*calibrated_Bx == NULL || *calibrated_By == NULL) {
        perror("Memory allocation failed");
        return;
    }

    for (int i = 0; i < length; i++) {
        (*calibrated_Bx)[i] = -Bx[i] / scale1;
        (*calibrated_By)[i] = -By[i] / scale2;
    }
}

void save_signals(double *HNS, double *HEW,
        TimeDomainFeatures* hns_features, TimeDomainFeatures* hew_features, Harmonic* harmonics,
        int length, const char *output_file) {
    FILE *f = fopen(output_file, "w");
     // Write the HNS time-domain features to the file (only values)
     if (hns_features!=NULL && hew_features!=NULL){
         fprintf(f, "%f\t%f\t%f\t%d\t%f\t%f\n",
                 hns_features->mean,
                 hns_features->std,
                 hns_features->rms,
                 hns_features->zcr,
                 hns_features->skewness,
                 hns_features->kurtosis);

         // Write the HEW time-domain features to the file (only values)
         fprintf(f, "%f\t%f\t%f\t%d\t%f\t%f\n",
                 hew_features->mean,
                 hew_features->std,
                 hew_features->rms,
                 hew_features->zcr,
                 hew_features->skewness,
                 hew_features->kurtosis);
    }
    // Write the harmonics values (only peak frequency and amplitude for each harmonic)
    // for (int h = 0; h < NUM_HARMONICS; h++) {
    //     fprintf(f, "%f\t%f\n", harmonics[h].peak_frequency, harmonics[h].amplitude);
    // }

    if (!f) {
        printf("filename: %s\n", output_file);
        perror("save_signals: File opening failed");
        return;
    }

    for (int i = 0; i < length; i++) {
        fprintf(f, "%d\t%d\n", (int) HNS[i], (int) HEW[i]);
    }

    fclose(f);
}

// Convert date to a serial date number
double datenum(int year, int month, int day, int hour, int min, int sec) {
    struct tm t = {0};
    t.tm_year = year - 1900;
    t.tm_mon = month - 1;
    t.tm_mday = day;
    t.tm_hour = hour;
    t.tm_min = min;
    t.tm_sec = sec;
    return (double)timegm(&t) / 86400.0 + 719529.0;
}

// Reads data from binary file into an SrdData structure
SrdData read_srd_file(const char *fpath) {
    SrdData data = {0, -1, 0, 0.0, NULL, NULL, 0, 0};
    uint64_t DATALOGGERID = 0xCAD0FFE51513FFDC;
    uint64_t ID;

    FILE *fp = fopen(fpath, "rb");
    if (fp == NULL) return data;

    fread(&ID, sizeof(uint64_t), 1, fp);
    if (ID != DATALOGGERID) {
        fprintf(stderr, "File \"%s\" is not a logger record!\n", fpath);
        fclose(fp);
        return data;
    }

    uint8_t S, MN, H, DAY, D, M, Y;
    fseek(fp, 8, SEEK_SET);
    fread(&S, sizeof(uint8_t), 1, fp);
    fread(&MN, sizeof(uint8_t), 1, fp);
    fread(&H, sizeof(uint8_t), 1, fp);
    fread(&DAY, sizeof(uint8_t), 1, fp);
    fread(&D, sizeof(uint8_t), 1, fp);
    fread(&M, sizeof(uint8_t), 1, fp);
    fread(&Y, sizeof(uint8_t), 1, fp);

    data.date = datenum(Y + 1970, M, D, H, MN, S);
    double t0 = datenum(2016, 1, 1, 0, 0, 0);
    double t1 = datenum(2017, 8, 1, 0, 0, 0);

    if (data.date > t0 && data.date < t1) {
        double tslop = 480.0 / 600.0;
        double dt = (data.date - t0) * (tslop / 86400.0);
        data.date -= dt;
    }

    fseek(fp, 15, SEEK_SET);
    float fs_value;
    fread(&fs_value, sizeof(float), 1, fp);
    data.fs = fs_value;

    fseek(fp, 19, SEEK_SET);
    uint8_t ch_value;
    fread(&ch_value, sizeof(uint8_t), 1, fp);
    data.ch = ch_value;

    fseek(fp, 20, SEEK_SET);
    float vbat_value;
    fread(&vbat_value, sizeof(float), 1, fp);
    data.vbat = vbat_value;

    data.ok = 1;

    fseek(fp, 512 + 16, SEEK_SET);

    uint16_t *temp_x = malloc(sizeof(uint16_t) * 1000000);
    int i = 0;
    while (fread(&temp_x[i], sizeof(uint16_t), 1, fp) == 1) {
        i++;
        if (i % 1000000 == 0) {
            temp_x = realloc(temp_x, sizeof(uint16_t) * (i + 1000000));
        }
    }
    fclose(fp);
    data.N = i;

    double MAX_VAL = (data.date < datenum(2017, 8, 10, 0, 0, 0)) ? 65535.0 : 32767.0;
    int faulty_shift = 0;
    for (int j = 0; j < data.N && j < 10000; j++) {
        if (temp_x[j] > MAX_VAL) {
            faulty_shift = 1;
            break;
        }
    }

    if (faulty_shift) {
        fp = fopen(fpath, "rb");
        fseek(fp, 512 + 17, SEEK_SET);
        i = 0;
        while (fread(&temp_x[i], sizeof(uint16_t), 1, fp) == 1) {
            i++;
        }
        fclose(fp);
        data.N = i;
    }

    if (data.N % 2 != 0) data.N--;

    data.x = malloc(sizeof(double) * data.N / (data.ch == 1 ? 2 : 1));
    if (data.ch == 1) data.y = malloc(sizeof(double) * data.N / 2);

    if (data.ch == 0) {
        for (int j = 0; j < data.N; j++) {
            data.x[j] = temp_x[j] * 4.096 / MAX_VAL - 2.048;
        }
    } else {
        for (int j = 0; j < data.N / 2; j++) {
            data.x[j] = temp_x[2 * j] * 4.096 / MAX_VAL - 2.048;
            data.y[j] = temp_x[2 * j + 1] * 4.096 / MAX_VAL - 2.048;
        }
    }

    free(temp_x);

    double mean_x = 0, mean_y = 0;
    for (int j = 0; j < data.N / (data.ch == 1 ? 2 : 1); j++) {
        mean_x += data.x[j];
        if (data.ch == 1) mean_y += data.y[j];
    }
    mean_x /= (data.N / (data.ch == 1 ? 2 : 1));
    if (data.ch == 1) mean_y /= (data.N / 2);

    for (int j = 0; j < data.N / (data.ch == 1 ? 2 : 1); j++) {
        data.x[j] -= mean_x;
        if (data.ch == 1) data.y[j] -= mean_y;
    }

    return data;
}

// Format the date as a string
void format_date(double dt, char *buffer, size_t buffer_size) {
    time_t raw_time = (time_t)((dt - 719529) * 86400);
    struct tm *timeinfo = gmtime(&raw_time);
    strftime(buffer, buffer_size, "%d %b %Y %H:%M:%S", timeinfo);
}

char* get_filename(double dt) {
    char *buffer = malloc(15); // Allocate memory for "YYYYMMDDHHMMSS\0"
    if (buffer == NULL) return NULL; // Check for allocation failure

    time_t raw_time = (time_t)((dt - 719529) * 86400);
    struct tm *timeinfo = gmtime(&raw_time);
    strftime(buffer, 15, "%Y%m%d%H%M%S", timeinfo);
    return buffer;
}
