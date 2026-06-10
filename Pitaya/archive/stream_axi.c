#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>
#include "rp.h"
#include "rp_hw-calib.h"

// AXI DMA SETTINGS
// 100,000 samples @ 125MS/s = 0.8 milliseconds of data (200KB)
#define TOTAL_SAMPLES 100000
#define PORT 5005

int main(int argc, char **argv) {
    char *server_ip = "169.254.118.176";
    if (argc > 1) {
        server_ip = argv[1];
    }

    if (rp_Init() != RP_OK) {
        fprintf(stderr, "Rp api init failed!\n");
        return -1;
    }
    
    rp_CalibrationReset(false, false);

    uint32_t axi_start, axi_size;
    if (rp_AcqAxiGetMemoryRegion(&axi_start, &axi_size) != RP_OK) {
        fprintf(stderr, "Failed to get AXI memory region\n");
        return -1;
    }

    // Initial Config
    rp_AcqReset();
    rp_AcqSetDecimation(RP_DEC_1);
    rp_AcqAxiSetDecimationFactor(RP_DEC_1);
    rp_AcqAxiSetTriggerDelay(RP_CH_1, TOTAL_SAMPLES);
    rp_AcqAxiSetBufferSamples(RP_CH_1, axi_start, TOTAL_SAMPLES);
    rp_AcqAxiEnable(RP_CH_1, true);

    // Setup Socket
    int sock = 0;
    struct sockaddr_in serv_addr;
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("\n Socket creation error \n");
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    if (inet_pton(AF_INET, server_ip, &serv_addr.sin_addr) <= 0) {
        printf("\nInvalid address/ Address not supported \n");
        return -1;
    }

    printf("Connecting to %s:%d...\n", server_ip, PORT);
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        printf("\nConnection Failed \n");
        return -1;
    }

    int16_t *big_buffer = (int16_t *)malloc(TOTAL_SAMPLES * sizeof(int16_t));

    printf("Starting continuous streaming (100k samples) to host...\n");
    int frame_count = 0;
    while (1) {
        struct timeval t0, t1, t2, t3;
        gettimeofday(&t0, NULL);

        // 3. Start Acquisition
        rp_AcqStart();
        rp_AcqSetTriggerSrc(RP_TRIG_SRC_NOW);

        // 4. Wait for the buffer to fill
        bool fill_state = false;
        while (!fill_state) {
            rp_AcqAxiGetBufferFillState(RP_CH_1, &fill_state);
        }
        rp_AcqStop();
        gettimeofday(&t1, NULL);

        // 5. Retrieve data
        uint32_t samples_read = TOTAL_SAMPLES;
        rp_AcqAxiGetDataRaw(RP_CH_1, 0, &samples_read, big_buffer);
        gettimeofday(&t2, NULL);

        // 6. Send data over socket
        int total_bytes = TOTAL_SAMPLES * sizeof(int16_t);
        int sent_bytes = 0;
        while (sent_bytes < total_bytes) {
            int n = send(sock, (char*)big_buffer + sent_bytes, total_bytes - sent_bytes, 0);
            if (n <= 0) {
                perror("send failed");
                goto cleanup;
            }
            sent_bytes += n;
        }
        gettimeofday(&t3, NULL);
        
        frame_count++;
        if (frame_count % 100 == 0) {
            long cap_ms = ((t1.tv_sec - t0.tv_sec) * 1000000 + (t1.tv_usec - t0.tv_usec)) / 1000;
            long cpy_ms = ((t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec)) / 1000;
            long net_ms = ((t3.tv_sec - t2.tv_sec) * 1000000 + (t3.tv_usec - t2.tv_usec)) / 1000;
            printf("Frame %d | Cap: %ldms | Cpy: %ldms | Net: %ldms\n", frame_count, cap_ms, cpy_ms, net_ms);
        }
    }

cleanup:
    free(big_buffer);
    close(sock);
    rp_AcqAxiEnable(RP_CH_1, false);
    rp_Release();
    return 0;
}
