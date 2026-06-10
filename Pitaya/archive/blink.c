#include <stdio.h>
#include <unistd.h>
#include "rp.h"

int main(int argc, char **argv) {
    // Initialize the library
    if (rp_Init() != RP_OK) {
        fprintf(stderr, "Red Pitaya API init failed!\n");
        return -1;
    }

    printf("Blinking LED 0 for 5 seconds...\n");

    for (int i = 0; i < 5; i++) {
        rp_DpinSetState(RP_LED0, RP_HIGH); // Turn LED on
        sleep(1);
        rp_DpinSetState(RP_LED0, RP_LOW);  // Turn LED off
        sleep(1);
    }

    // Release resources
    rp_Release();
    return 0;
}
