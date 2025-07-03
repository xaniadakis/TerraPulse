#ifndef LOG_H
#define LOG_H

#include <stdio.h>

extern FILE *log_file;  // Declaration, not a definition

// #define LOG(...) fprintf(log_file, __VA_ARGS__)
#define LOG(...) do { printf(__VA_ARGS__); fflush(stdout); } while (0)

#endif // LOG_H
