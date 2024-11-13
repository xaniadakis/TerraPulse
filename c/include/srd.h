//
// Created by vag on 10/15/24.
//

#ifndef SRD_H
#define SRD_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    double date;
    double fs;
    uint8_t ch;
    float vbat;
    double *x;
    double *y;
    int N;
    int ok;
} SrdData;

#endif //SRD_H
