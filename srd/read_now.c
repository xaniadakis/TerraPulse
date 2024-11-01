#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    double date;
    double fs;
    uint8_t ch;
    float vbat;
    int ok;
} SrdInfo;

double datenum(int year, int month, int day, int hour, int min, int sec) {
    struct tm t = {0};
    t.tm_year = year - 1900;
    t.tm_mon = month - 1;
    t.tm_mday = day;
    t.tm_hour = hour;
    t.tm_min = min;
    t.tm_sec = sec;
    return (double)mktime(&t) / 86400.0 + 719529.0;
}

void format_date(double dt, char *buffer, size_t buffer_size) {
    time_t raw_time = (time_t)((dt - 719529) * 86400);  // Convert MATLAB datenum to Unix time
    struct tm *timeinfo = gmtime(&raw_time);  // Convert to UTC time structure

    strftime(buffer, buffer_size, "%d %b %Y %H:%M:%S", timeinfo);  // Format the date
}

SrdInfo get_srd_info(const char *fname) {
    SrdInfo info = {0, -1, 0, 0.0, 0};
    uint64_t DATALOGGERID = 0xCAD0FFE51513FFDC;
    uint64_t ID;

    FILE *fp = fopen(fname, "rb");
    if (fp == NULL) return info;

    fseek(fp, 0, SEEK_END);
    if (ftell(fp) < 2 * 512) {
        fclose(fp);
        return info;
    }
    fseek(fp, 0, SEEK_SET);

    fread(&ID, sizeof(uint64_t), 1, fp);
    if (ID != DATALOGGERID) {
        fprintf(stderr, "File \"%s\" is not logger record!\n", fname);
        fclose(fp);
        return info;
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

    info.date = datenum(Y + 1970, M, D, H, MN, S);
    double t0 = datenum(2016, 1, 1, 0, 0, 0);
    double t1 = datenum(2017, 8, 1, 0, 0, 0);

    if (info.date > t0 && info.date < t1) {
        double tslop = 480.0 / 600.0;
        double dt = (info.date - t0) * (tslop / 86400.0);
        info.date -= dt;
    }

    fseek(fp, 15, SEEK_SET);
    fread(&info.fs, sizeof(float), 1, fp);
    fseek(fp, 19, SEEK_SET);
    fread(&info.ch, sizeof(uint8_t), 1, fp);
    fseek(fp, 20, SEEK_SET);
    fread(&info.vbat, sizeof(float), 1, fp);

    info.ok = 1;
    fclose(fp);
    return info;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    SrdInfo info = get_srd_info(argv[1]);
    if (info.ok) {
        char date_str[30];
        format_date(info.date, date_str, sizeof(date_str));
        printf("Date: %s\n", date_str);
        printf("Sample Rate: %lf\n", info.fs);
        printf("Channel: %d\n", info.ch);
        printf("Battery Voltage: %f\n", info.vbat);
    } else {
        printf("Failed to read SRD info.\n");
    }

    return 0;
}
