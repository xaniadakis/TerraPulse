#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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
SrdData get_srd_data(const char *fpath) {
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

// Downsample signal by averaging and save to a file
void downsample_signal(const double *input, int input_size, int downsample_factor, const char *filename) {
    int new_size = input_size / downsample_factor;
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Failed to open file \"%s\" for writing\n", filename);
        return;
    }
    for (int i = 0; i < new_size; i++) {
        double sum = 0.0;
        for (int j = 0; j < downsample_factor; j++) {
            sum += input[i * downsample_factor + j];
        }
        double avg = sum / downsample_factor;
        fprintf(file, "%lf\n", avg);
    }
    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <filepath>\n", argv[0]);
        return 1;
    }

    clock_t start_time = clock();

    SrdData data = get_srd_data(argv[1]);

    int downsample_factor = (int)(data.fs / 100);  // Calculate downsample factor for 100 Hz
    printf("%dx downsampling!\n", downsample_factor);
    downsample_signal(data.x, data.N, downsample_factor, "downsampled_signal.txt");

    clock_t end_time = clock();
    double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Time taken to read & downsample file: %f seconds\n", time_spent);

    if (data.ok) {
        char date_str[30];
        format_date(data.date, date_str, sizeof(date_str));
        printf("Date: %s\n", date_str);
        printf("Sample Rate: %lf\n", data.fs);
        printf("Number of samples: %d\n", data.N);
        printf("Channel Count: %s\n", data.ch == 0 ? "Single" : "Dual");
        printf("Battery Voltage: %e V\n", data.vbat);
    } else {
        printf("Failed to load data.\n");
    }

    free(data.x);
    if (data.ch == 1) free(data.y);

    return 0;
}
