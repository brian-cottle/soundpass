#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include "rp.h"
#include "rp_hw-calib.h"

int main(int argc, char **argv) {
    // 1. Initialize API
    if (rp_Init() != RP_OK) {
        fprintf(stderr, "Rp api init failed!\n");
        return -1;
    }

    // 2. FIX: Reset calibration to neutral defaults (Gain=1, Offset=0)
    // This bypasses the "divide by zero" errors on your board.
    printf("Resetting calibration to neutral defaults...\n");
    rp_CalibrationReset(false, false);

    uint32_t buff_size = 16384;
    float *buff = (float *)malloc(buff_size * sizeof(float));

    rp_AcqReset();
    rp_AcqSetDecimation(RP_DEC_1);

    // Now that calibration is reset, this 0.1V trigger will work correctly!
    rp_AcqSetTriggerLevel(RP_T_CH_1, 0.1); 
    
    rp_AcqStart();
    usleep(100);

    printf("Waiting for trigger on IN1 (Level: 0.1V)...\n");
    rp_AcqSetTriggerSrc(RP_TRIG_SRC_CHA_PE);

    rp_acq_trig_state_t state = RP_TRIG_STATE_WAITING;
    while(1) {
        rp_AcqGetTriggerState(&state);
        if(state == RP_TRIG_STATE_TRIGGERED) break;
        usleep(10);
    }

    // Get the data in Volts
    rp_AcqGetOldestDataV(RP_CH_1, &buff_size, buff);

    printf("Triggered! First 20 samples (Volts):\n");
    for(int i = 0; i < 20; i++) {
        printf("[%d]: %f V\n", i, buff[i]);
    }

    free(buff);
    rp_Release();
    return 0;
}
