
#include "io.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int read_dat_file(const char *fn, int **Bx, int **By, int *nr_out) {
    FILE *f = fopen(fn, "rb");
    if (!f) {
        perror("read_dat_file: File opening failed");
        return 1;
    }

    fseek(f, 64, SEEK_SET);
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

    for (int j = 0; j < 89; j++) {
        if (i + 4 >= file_size) {  // Check bounds before accessing data[i + 3] and data[i + 4]
            fprintf(stderr, "Out of bounds access in read_dat_file (loop 1) for file: %s\n", fn);
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

    for (int n = 0; n < 8836; n++) {
        for (int j = 0; j < 102; j++) {
            if (i + 4 >= file_size) {  // Check bounds before accessing data[i + 3] and data[i + 4]
                fprintf(stderr, "Out of bounds access in read_dat_file (loop 2) for file: %s\n", fn);
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

    for (int j = 0; j < 82; j++) {
        if (i + 4 >= file_size) {  // Check bounds before accessing data[i + 3] and data[i + 4]
            fprintf(stderr, "Out of bounds access in read_dat_file (loop 3) for file: %s\n", fn);
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

    // Adjust for final array values
    while ((*Bx)[nr] == 0 || (*By)[nr] == 0) {
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
