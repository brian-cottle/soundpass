#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/time.h>
#include "rp.h"

// AXI DMA SETTINGS
// 1,000,000 samples @ 125MS/s = 8 milliseconds of continuous data (2MB)
#define TOTAL_SAMPLES 1000000

int main(int argc, char **argv) {
    if (rp_Init() != RP_OK) {
        fprintf(stderr, "Rp api init failed!\n");
        return -1;
    }

    uint32_t axi_start, axi_size;
    if (rp_AcqAxiGetMemoryRegion(&axi_start, &axi_size) != RP_OK) {
        fprintf(stderr, "Failed to get AXI memory region\n");
        return -1;
    }

    // 1. Configure AXI
    rp_AcqReset();
    rp_AcqSetDecimation(RP_DEC_1);
    rp_AcqAxiSetDecimationFactor(RP_DEC_1);
    rp_AcqAxiSetTriggerDelay(RP_CH_1, TOTAL_SAMPLES);
    
    if (rp_AcqAxiSetBufferSamples(RP_CH_1, axi_start, TOTAL_SAMPLES) != RP_OK) {
        fprintf(stderr, "Failed to set AXI buffer samples\n");
        return -1;
    }
    rp_AcqAxiEnable(RP_CH_1, true);

    struct timeval t0, t1, t2;
    
    printf("Starting MAX SPEED capture of 2MB raw stream...\n");
    gettimeofday(&t0, NULL);

    // 2. Start Acquisition
    rp_AcqStart();
    rp_AcqSetTriggerSrc(RP_TRIG_SRC_NOW); // Trigger immediately

    // 3. Wait for the buffer to fill
    bool fill_state = false;
    while (!fill_state) {
        rp_AcqAxiGetBufferFillState(RP_CH_1, &fill_state);
    }
    rp_AcqStop();

    gettimeofday(&t1, NULL); // Stop timer for hardware capture

    // 4. Retrieve the giant data block
    int16_t *big_buffer = (int16_t *)malloc(TOTAL_SAMPLES * sizeof(int16_t));
    uint32_t samples_read = TOTAL_SAMPLES;
    rp_AcqAxiGetDataRaw(RP_CH_1, 0, &samples_read, big_buffer);

    // 5. Raw Binary Dump (No CSV, No Math, No Pulse Extraction)
    FILE *fp = fopen("data_raw.bin", "wb");
    if (fp != NULL) {
        fwrite(big_buffer, sizeof(int16_t), TOTAL_SAMPLES, fp);
        fclose(fp);
    } else {
        fprintf(stderr, "Failed to open data_raw.bin for writing!\n");
    }

    gettimeofday(&t2, NULL); // Stop timer for save

    // 6. Calculate Timings
    long capture_micros = ((t1.tv_sec - t0.tv_sec) * 1000000) + (t1.tv_usec - t0.tv_usec);
    long save_micros = ((t2.tv_sec - t1.tv_sec) * 1000000) + (t2.tv_usec - t1.tv_usec);
    long total_micros = ((t2.tv_sec - t0.tv_sec) * 1000000) + (t2.tv_usec - t0.tv_usec);

    printf("----------------------------------------\n");
    printf("Hardware Capture Time:  %.2f ms\n", capture_micros / 1000.0);
    printf("RAM-to-Disk Save Time:  %.2f ms\n", save_micros / 1000.0);
    printf("Total Execution Time:   %.2f ms\n", total_micros / 1000.0);
    printf("----------------------------------------\n");

    free(big_buffer);
    rp_AcqAxiEnable(RP_CH_1, false);
    rp_Release();
    return 0;
}