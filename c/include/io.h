//
// Created by vag on 10/15/24.
//

#ifndef MYIO_H
#define MYIO_H

void read_dat_file(const char *fn, int **Bx, int **By, int *nr_out);
void calibrate_HYL(int *Bx, int *By, int length, double **calibrated_Bx, double **calibrated_By);
void save_signals(double *HNS, double *HEW, int length, const char *output_file);

#endif //MYIO_H
