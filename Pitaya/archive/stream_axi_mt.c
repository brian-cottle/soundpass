#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
#include "rp.h"
#include "rp_hw-calib.h"

#define TOTAL_SAMPLES 1000000
#define PORT 5005
#define NUM_BUFFERS 2
#define PRE_TRIGGER 100

int pulse_len = 6000;

// Shared state
int16_t *buffers[NUM_BUFFERS];
bool buffer_ready[NUM_BUFFERS] = {false, false};
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
bool keep_running = true;

// Processing params
int threshold_counts = 20; // Default: ~50mV (20 counts)
int holdoff_samples = 6000; // Wait for the whole pulse to finish (updated dynamically)

int sock = 0;

void *processing_thread(void *arg) {
    int read_idx = 0;
    int32_t *sum_pulse = (int32_t *)malloc(pulse_len * sizeof(int32_t));
    float *send_buf = (float *)malloc((pulse_len + 1) * sizeof(float));

    printf("Processing & Network thread started.\n");

    while (keep_running) {
        pthread_mutex_lock(&lock);
        while (!buffer_ready[read_idx] && keep_running) {
            pthread_cond_wait(&cond, &lock);
        }
        if (!keep_running) {
            pthread_mutex_unlock(&lock);
            break;
        }
        pthread_mutex_unlock(&lock);

        // We have a buffer ready!
        int16_t *buf = buffers[read_idx];
        
        memset(sum_pulse, 0, pulse_len * sizeof(int32_t));
        int num_pulses = 0;
        int i = 1000; // Skip edge effects
        
        // Fast Pulse Extraction Loop
        while (i < TOTAL_SAMPLES - pulse_len && num_pulses < 1000) {
            if (abs(buf[i]) > threshold_counts) {
                int start_idx = i - PRE_TRIGGER;
                if (start_idx < 0) start_idx = 0;
                
                for (int j = 0; j < pulse_len; j++) {
                    sum_pulse[j] += buf[start_idx + j];
                }
                num_pulses++;
                i += holdoff_samples; // Skip over the ringing
            } else {
                i += 5; // Coarse search to speed up the loop
            }
        }

        // Prepare frame for transmission
        send_buf[0] = (float)num_pulses;

        if (num_pulses > 0) {
            for (int j = 0; j < pulse_len; j++) {
                // Average and convert to Volts simultaneously
                send_buf[j + 1] = ((float)sum_pulse[j] / num_pulses) * (20.0 / 8192.0);
            }
        } else {
            for (int j = 0; j < pulse_len; j++) {
                send_buf[j + 1] = 0.0f;
            }
        }

        // Send the packet over the network
        int total_bytes = (pulse_len + 1) * sizeof(float);
        int sent_bytes = 0;
        while (sent_bytes < total_bytes) {
            int n = send(sock, (char*)send_buf + sent_bytes, total_bytes - sent_bytes, 0);
            if (n <= 0) {
                perror("Socket send failed! Host disconnected.");
                keep_running = false;
                break;
            }
            sent_bytes += n;
        }

        // Mark buffer as free for the capture thread
        pthread_mutex_lock(&lock);
        buffer_ready[read_idx] = false;
        pthread_mutex_unlock(&lock);

        read_idx = (read_idx + 1) % NUM_BUFFERS;
    }
    return NULL;
}

int main(int argc, char **argv) {
    char *server_ip = "169.254.118.176";
    if (argc > 1) server_ip = argv[1];
    if (argc > 2) threshold_counts = atoi(argv[2]); // Allow dynamic thresholding from CLI
    if (argc > 3) {
        pulse_len = atoi(argv[3]);
        holdoff_samples = pulse_len;
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

    int mem_fd;
    if ((mem_fd = open("/dev/mem", O_RDWR | O_SYNC)) < 0) {
        perror("Failed to open /dev/mem");
        return -1;
    }

    void *mmap_ptr = mmap(NULL, axi_size, PROT_READ, MAP_SHARED, mem_fd, axi_start);
    if (mmap_ptr == MAP_FAILED) {
        perror("mmap failed");
        return -1;
    }

    rp_AcqReset();
    rp_AcqSetDecimation(RP_DEC_1);
    rp_AcqAxiSetDecimationFactor(RP_DEC_1);
    rp_AcqAxiSetTriggerDelay(RP_CH_1, TOTAL_SAMPLES);
    rp_AcqAxiSetBufferSamples(RP_CH_1, axi_start, TOTAL_SAMPLES);
    rp_AcqAxiEnable(RP_CH_1, true);

    struct sockaddr_in serv_addr;
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("\n Socket creation error \n");
        return -1;
    }
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
    if (inet_pton(AF_INET, server_ip, &serv_addr.sin_addr) <= 0) {
        printf("\nInvalid address\n");
        return -1;
    }
    printf("Connecting to Host at %s:%d...\n", server_ip, PORT);
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        printf("\nConnection Failed! Make sure Python receiver is running.\n");
        return -1;
    }

    // Allocate memory for the ping-pong buffers
    for (int i = 0; i < NUM_BUFFERS; i++) {
        buffers[i] = (int16_t *)malloc(TOTAL_SAMPLES * sizeof(int16_t));
    }

    // Start the extraction and networking thread
    pthread_t p_thread;
    pthread_create(&p_thread, NULL, processing_thread, NULL);

    printf("Starting MULTI-THREADED continuous capture (100k samples)...\n");
    int write_idx = 0;
    int frame_count = 0;

    // Capture Loop
    while (keep_running) {
        struct timeval t0, t1, t2;
        gettimeofday(&t0, NULL);

        // Wait until the target buffer is free (consumed by the other thread)
        bool can_write = false;
        while (!can_write && keep_running) {
            pthread_mutex_lock(&lock);
            if (!buffer_ready[write_idx]) {
                can_write = true;
            }
            pthread_mutex_unlock(&lock);
            if (!can_write) usleep(100); // tiny sleep to prevent locking up CPU
        }
        if (!keep_running) break;

        // Hardware Capture
        rp_AcqStart();
        rp_AcqSetTriggerSrc(RP_TRIG_SRC_NOW);

        bool fill_state = false;
        while (!fill_state && keep_running) {
            rp_AcqAxiGetBufferFillState(RP_CH_1, &fill_state);
        }
        rp_AcqStop();
        gettimeofday(&t1, NULL);

        // DMA Memory Copy via mmap
        memcpy(buffers[write_idx], mmap_ptr, TOTAL_SAMPLES * sizeof(int16_t));
        
        // Signal the processing thread that a new buffer is fully copied and ready
        pthread_mutex_lock(&lock);
        buffer_ready[write_idx] = true;
        pthread_cond_signal(&cond);
        pthread_mutex_unlock(&lock);

        gettimeofday(&t2, NULL);

        write_idx = (write_idx + 1) % NUM_BUFFERS;
        frame_count++;

        if (frame_count % 100 == 0) {
            long cap_ms = ((t1.tv_sec - t0.tv_sec) * 1000000 + (t1.tv_usec - t0.tv_usec)) / 1000;
            long cpy_ms = ((t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec)) / 1000;
            printf("Capture %d | Hardware: %ldms | DMA Copy: %ldms\n", frame_count, cap_ms, cpy_ms);
        }
    }

    // Cleanup
    keep_running = false;
    pthread_cond_signal(&cond);
    pthread_join(p_thread, NULL);

    for (int i = 0; i < NUM_BUFFERS; i++) free(buffers[i]);
    close(sock);
    rp_AcqAxiEnable(RP_CH_1, false);
    rp_Release();
    return 0;
}