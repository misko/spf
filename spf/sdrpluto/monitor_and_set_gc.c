#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#define GAIN_CH0_PATH "/sys/bus/iio/devices/iio:device0/in_voltage0_hardwaregain"
#define GAIN_CH1_PATH "/sys/bus/iio/devices/iio:device0/in_voltage1_hardwaregain"

int main(void)
{
    // Keep track of the last value we wrote to channel 1
    char last_written_gain[32];
    last_written_gain[0] = '\0'; // start empty

    while (1) {
        // 1) Read gain from channel 0
        FILE *f0 = fopen(GAIN_CH0_PATH, "r");
        if (!f0) {
            fprintf(stderr, "Failed to open %s: %s\n", GAIN_CH0_PATH, strerror(errno));
            return 1;
        }

        char buf[32];
        if (!fgets(buf, sizeof(buf), f0)) {
            fprintf(stderr, "Failed to read from %s: %s\n", GAIN_CH0_PATH, strerror(errno));
            fclose(f0);
            return 1;
        }
        fclose(f0);

        // Optional: remove trailing newline
        // (Typically sysfs will have a newline at the end)
        size_t len = strlen(buf);
        if (len > 0 && buf[len - 1] == '\n') {
            buf[len - 1] = '\0';
        }

        // 2) Compare with our last written gain to channel 1
        if (strcmp(buf, last_written_gain) != 0) {
            // Gains differ, so we need to write the new gain to channel 1
            FILE *f1 = fopen(GAIN_CH1_PATH, "w");
            if (!f1) {
                fprintf(stderr, "Failed to open %s for writing: %s\n", 
                        GAIN_CH1_PATH, strerror(errno));
                return 1;
            }

            // 3) Write the new gain to channel 1
            if (fprintf(f1, "%s\n", buf) < 0) {
                fprintf(stderr, "Failed to write to %s: %s\n", GAIN_CH1_PATH, strerror(errno));
                fclose(f1);
                return 1;
            }
            fclose(f1);

            // 4) Update our "last_written_gain" tracker
            strncpy(last_written_gain, buf, sizeof(last_written_gain) - 1);
            last_written_gain[sizeof(last_written_gain) - 1] = '\0';
        }

        // Sleep e.g. 0.1 seconds (10 Hz)
        usleep(100000);
    }

    return 0;
}

