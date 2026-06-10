#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/time.h>
#include "rp.h"

// AXI DMA SETTINGS - CALIBRATED FOR 2MB LIMIT
// 1,000,000 samples @ 125MS/s = 8 milliseconds of continuous data
// This fits in 2,000,000 bytes (2MB)
#define TOTAL_SAMPLES 1000000
#define PULSE_SIZE 7500
#define NUM_PULSES 80
// In HV mode (±20V scale), 8192 counts = 20V. 
// So 100mV is roughly 41 counts. We use 40 as our threshold.
#define TRIGGER_SENSITIVITY 40 

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

    printf("AXI DMA Initialized at 0x%X (Available Size: %u bytes)\n", axi_start, axi_size);

    // 1. Configure AXI
    rp_AcqReset();
    rp_AcqSetDecimation(RP_DEC_1);
    rp_AcqAxiSetDecimationFactor(RP_DEC_1);
    
    // CRITICAL: Tell AXI to capture exactly TOTAL_SAMPLES *after* the trigger
    rp_AcqAxiSetTriggerDelay(RP_CH_1, TOTAL_SAMPLES);
    
    // Map Channel 1 - we use TOTAL_SAMPLES which is 1,000,000 (2MB)
    if (rp_AcqAxiSetBufferSamples(RP_CH_1, axi_start, TOTAL_SAMPLES) != RP_OK) {
        fprintf(stderr, "Failed to set AXI buffer samples\n");
        return -1;
    }
    rp_AcqAxiEnable(RP_CH_1, true);

    // 2. Start Acquisition
    printf("Starting CONTINUOUS capture of 8ms stream (125 MS/s)...\n");
    struct timeval start, end;
    gettimeofday(&start, NULL);

    rp_AcqStart();
    rp_AcqSetTriggerSrc(RP_TRIG_SRC_NOW); // Trigger immediately to fill the reserved RAM

    // 3. Wait for the buffer to fill
    bool fill_state = false;
    while (!fill_state) {
        rp_AcqAxiGetBufferFillState(RP_CH_1, &fill_state);
        usleep(500); 
    }
    rp_AcqStop();

    gettimeofday(&end, NULL);
    long micros = ((end.tv_sec - start.tv_sec) * 1000000) + (end.tv_usec - start.tv_usec);
    printf("Capture finished in %.2f ms.\n", micros / 1000.0);

    // 4. Retrieve the giant data block
    int16_t *big_buffer = (int16_t *)malloc(TOTAL_SAMPLES * sizeof(int16_t));
    uint32_t samples_read = TOTAL_SAMPLES;
    rp_AcqAxiGetDataRaw(RP_CH_1, 0, &samples_read, big_buffer);

    // 5. Software Pulse Extraction
    printf("Extracting pulses from stream...\n");
    int16_t **pulses = (int16_t **)malloc(NUM_PULSES * sizeof(int16_t *));
    int pulse_count = 0;
    int i = 1000; // Start after edge effects
    
    int16_t baseline = big_buffer[100]; // Get baseline from start of buffer

    while (i < TOTAL_SAMPLES - PULSE_SIZE && pulse_count < NUM_PULSES) {
        // Look for a jump relative to local baseline (Checking both positive and negative jumps)
        if (big_buffer[i] > (baseline + TRIGGER_SENSITIVITY) || big_buffer[i] < (baseline - TRIGGER_SENSITIVITY)) {
            pulses[pulse_count] = (int16_t *)malloc(PULSE_SIZE * sizeof(int16_t));
            // Take 500 samples before and 7000 after
            for (int j = 0; j < PULSE_SIZE; j++) {
                pulses[pulse_count][j] = big_buffer[i - 500 + j];
            }
            pulse_count++;
            i += 10000; // Jump 80us ahead to skip the current pulse
            
            // Re-sample baseline for next pulse
            if (i < TOTAL_SAMPLES) baseline = big_buffer[i-100];
        } else {
            i++;
        }
    }

    printf("Successfully extracted %d pulses.\n", pulse_count);

    // 6. Write to CSV
    if (pulse_count > 0) {
        printf("Writing to data_axi.csv...\n");
        FILE *fp = fopen("data_axi.csv", "w");
        for (int c = 0; c < pulse_count; c++) {
            fprintf(fp, "Pulse_%d%s", c + 1, (c == pulse_count - 1) ? "" : ",");
        }
        fprintf(fp, "\n");

        for (int j = 0; j < PULSE_SIZE; j++) {
            for (int c = 0; c < pulse_count; c++) {
                // In HV mode: 8192 counts = 20V
                fprintf(fp, "%f%s", (float)pulses[c][j] * 20.0 / 8192.0, (c == pulse_count - 1) ? "" : ",");
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        printf("Done!\n");
    } else {
        printf("No pulses found! The signal might be smaller than 100mV.\n");
    }

    // Cleanup
    for (int c = 0; c < pulse_count; c++) free(pulses[c]);
    free(pulses);
    free(big_buffer);
    rp_AcqAxiEnable(RP_CH_1, false);
    rp_Release();
    return 0;
}