//
// Created by vag on 10/15/24.
//

#ifndef MYIO_H
#define MYIO_H

#include <feature.h>
#include <signanalysis.h>
#include "srd.h"

int read_dat_file(const char *fn, int **Bx, int **By, int *nr_out);
void calibrate_HYL(int *Bx, int *By, int length, const char* date, double **calibrated_Bx, double **calibrated_By);
void save_signals(double *HNS, double *HEW, TimeDomainFeatures* hns_features, TimeDomainFeatures* hew_features, Harmonic* harmonics, int length, const char *output_file);

double datenum(int year, int month, int day, int hour, int min, int sec);
SrdData read_srd_file(const char *fpath);
void format_date(double dt, char *buffer, size_t buffer_size);
char* get_filename(double dt);

#endif //MYIO_H
