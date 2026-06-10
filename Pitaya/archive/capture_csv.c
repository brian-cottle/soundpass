#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/time.h>
#include "rp.h"
#include "rp_hw-calib.h"

// ULTRA-FAST SETTINGS
#define BUF_SIZE 7500      
#define NUM_CAPTURES 200    
#define TRIGGER_DELAY 7692  

int main(int argc, char **argv) {
    if (rp_Init() != RP_OK) {
        fprintf(stderr, "Rp api init failed!\n");
        return -1;
    }

    rp_CalibrationReset(false, false);

    // Use INT16 to avoid slow floating point math in the loop
    int16_t **buffers = (int16_t **)malloc(NUM_CAPTURES * sizeof(int16_t *));
    for (int i = 0; i < NUM_CAPTURES; i++) {
        buffers[i] = (int16_t *)malloc(BUF_SIZE * sizeof(int16_t));
    }

    rp_AcqReset();
    rp_AcqSetDecimation(RP_DEC_1);
    rp_AcqSetTriggerLevel(RP_T_CH_1, 0.1); 
    rp_AcqSetTriggerDelay(TRIGGER_DELAY);

    printf("Starting ULTRA-FAST capture of %d pulses...\n", NUM_CAPTURES);

    struct timeval start, end;
    rp_acq_trig_state_t state;
    uint32_t size;
    
    gettimeofday(&start, NULL);

    for (int c = 0; c < NUM_CAPTURES; c++) {
        rp_AcqStart();
        
        // Set trigger source immediately
        rp_AcqSetTriggerSrc(RP_TRIG_SRC_CHA_PE);

        // Fast poll for trigger
        while(1) {
            rp_AcqGetTriggerState(&state);
            if(state == RP_TRIG_STATE_TRIGGERED) break;
        }

        // To catch 10kHz, we skip the slow 'GetBufferFillState' 
        // The hardware will handle the wrapping.
        size = BUF_SIZE;
        rp_AcqGetOldestDataRaw(RP_CH_1, &size, buffers[c]);
    }

    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    
    printf("Successfully captured %d pulses in %.2f milliseconds.\n", NUM_CAPTURES, micros / 1000.0);
    printf("Actual Pulse Acquisition Rate: %.2f kHz\n", (float)NUM_CAPTURES / (micros / 1000000.0) / 1000.0);

    printf("Writing to data.csv (Converting to Volts now)...\n");
    FILE *fp = fopen("data.csv", "w");
    
    // Write header
    for (int c = 0; c < NUM_CAPTURES; c++) {
        fprintf(fp, "Pulse_%d%s", c + 1, (c == NUM_CAPTURES - 1) ? "" : ",");
    }
    fprintf(fp, "\n");

    // Write data rows with late-stage conversion to Volts
    // For LV mode, 8192 counts = 1.0V.
    for (int i = 0; i < BUF_SIZE; i++) {
        for (int c = 0; c < NUM_CAPTURES; c++) {
            float volts = (float)buffers[c][i] / 8192.0;
            fprintf(fp, "%f%s", volts, (c == NUM_CAPTURES - 1) ? "" : ",");
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    printf("Done!\n");

    for (int i = 0; i < NUM_CAPTURES; i++) free(buffers[i]);
    free(buffers);
    rp_Release();
    return 0;
}
