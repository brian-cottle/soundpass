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
#include <sys/ioctl.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>
#include <signal.h>
#include "rp.h"
#include "rp_hw-calib.h"

// AXI SETTINGS
#define TOTAL_SAMPLES 1000000
#define PORT 5005
#define NUM_BUFFERS 2
#define PRE_TRIGGER 100

// BNO055 SETTINGS
#define BNO055_ADDRESS_A    0x28

// Shared state
int16_t *buffers[NUM_BUFFERS];
bool buffer_ready[NUM_BUFFERS] = {false, false};
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
volatile bool keep_running = true;

int pulse_len = 6000;
int threshold_counts = 20; 
int holdoff_samples = 6000;
int buffers_to_average = 1;

int sock = 0;
struct sockaddr_in serv_addr;
int i2c_fd = 0;
int imu_error_count = 0;

// Thread-safety and shared IMU structures
pthread_mutex_t i2c_lock = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    float qw, qx, qy, qz;
    uint8_t sys_cal, gyro_cal, accel_cal, mag_cal;
} imu_data_t;

imu_data_t shared_imu_data = {1.0f, 0.0f, 0.0f, 0.0f, 0, 0, 0, 0};
pthread_mutex_t imu_lock = PTHREAD_MUTEX_INITIALIZER;

// --- BNO055 HAL WRAPPERS ---
#include "bno055.h"

struct bno055_t bno;

s8 bno055_i2c_bus_write(u8 dev_addr, u8 reg_addr, u8 *reg_data, u8 cnt) {
    pthread_mutex_lock(&i2c_lock);
    u8 array[256];
    array[0] = reg_addr;
    for (u8 i = 0; i < cnt; i++) {
        array[i + 1] = reg_data[i];
    }
    if (write(i2c_fd, array, cnt + 1) != cnt + 1) {
        pthread_mutex_unlock(&i2c_lock);
        return BNO055_ERROR;
    }
    pthread_mutex_unlock(&i2c_lock);
    return BNO055_SUCCESS;
}

s8 bno055_i2c_bus_read(u8 dev_addr, u8 reg_addr, u8 *reg_data, u8 cnt) {
    pthread_mutex_lock(&i2c_lock);
    struct i2c_msg msgs[2];
    struct i2c_rdwr_ioctl_data msgset[1];

    msgs[0].addr = dev_addr;
    msgs[0].flags = 0; // Write flag
    msgs[0].len = 1;
    msgs[0].buf = &reg_addr;

    msgs[1].addr = dev_addr;
    msgs[1].flags = I2C_M_RD; // Read flag (forces repeated start)
    msgs[1].len = cnt;
    msgs[1].buf = reg_data;

    msgset[0].msgs = msgs;
    msgset[0].nmsgs = 2;

    if (ioctl(i2c_fd, I2C_RDWR, &msgset) < 0) {
        pthread_mutex_unlock(&i2c_lock);
        return BNO055_ERROR;
    }
    pthread_mutex_unlock(&i2c_lock);
    return BNO055_SUCCESS;
}

void bno055_delay_msek(u32 msek) {
    usleep(msek * 1000);
}

// --- THREADS ---

double get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
}

s8 read_quaternion_sandwich(struct bno055_quaternion_t *quat) {
    u8 data_w1[2], data_x[2], data_y[2], data_z[2], data_w2[2];
    s8 r_w1, r_x, r_y, r_z, r_w2;
    
    for (int retry = 0; retry < 5; retry++) {
        // Read W1 (1 byte LSB, then 1 byte MSB)
        r_w1  = bno055_i2c_bus_read(BNO055_ADDRESS_A, BNO055_QUATERNION_DATA_W_LSB_ADDR, &data_w1[0], 1);
        r_w1 |= bno055_i2c_bus_read(BNO055_ADDRESS_A, BNO055_QUATERNION_DATA_W_MSB_ADDR, &data_w1[1], 1);
        
        // Read X
        r_x  = bno055_i2c_bus_read(BNO055_ADDRESS_A, BNO055_QUATERNION_DATA_X_LSB_ADDR, &data_x[0], 1);
        r_x |= bno055_i2c_bus_read(BNO055_ADDRESS_A, BNO055_QUATERNION_DATA_X_MSB_ADDR, &data_x[1], 1);
        
        // Read Y
        r_y  = bno055_i2c_bus_read(BNO055_ADDRESS_A, BNO055_QUATERNION_DATA_Y_LSB_ADDR, &data_y[0], 1);
        r_y |= bno055_i2c_bus_read(BNO055_ADDRESS_A, BNO055_QUATERNION_DATA_Y_MSB_ADDR, &data_y[1], 1);
        
        // Read Z
        r_z  = bno055_i2c_bus_read(BNO055_ADDRESS_A, BNO055_QUATERNION_DATA_Z_LSB_ADDR, &data_z[0], 1);
        r_z |= bno055_i2c_bus_read(BNO055_ADDRESS_A, BNO055_QUATERNION_DATA_Z_MSB_ADDR, &data_z[1], 1);
        
        // Read W2
        r_w2  = bno055_i2c_bus_read(BNO055_ADDRESS_A, BNO055_QUATERNION_DATA_W_LSB_ADDR, &data_w2[0], 1);
        r_w2 |= bno055_i2c_bus_read(BNO055_ADDRESS_A, BNO055_QUATERNION_DATA_W_MSB_ADDR, &data_w2[1], 1);
        
        s16 w1 = (s16)(((uint16_t)data_w1[1] << 8) | data_w1[0]);
        s16 w2 = (s16)(((uint16_t)data_w2[1] << 8) | data_w2[0]);
        
        if (w1 == w2 && r_w1 == BNO055_SUCCESS && r_x == BNO055_SUCCESS && r_y == BNO055_SUCCESS && r_z == BNO055_SUCCESS && r_w2 == BNO055_SUCCESS) {
            quat->w = w1;
            quat->x = (s16)(((uint16_t)data_x[1] << 8) | data_x[0]);
            quat->y = (s16)(((uint16_t)data_y[1] << 8) | data_y[0]);
            quat->z = (s16)(((uint16_t)data_z[1] << 8) | data_z[0]);
            return BNO055_SUCCESS;
        }
    }
    return BNO055_ERROR;
}

void handle_sigint(int sig) {
    printf("\nCaptured Ctrl+C/termination signal. Stopping cleanly...\n");
    pthread_mutex_lock(&lock);
    keep_running = false;
    pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&lock);
}

void reinit_imu() {
    printf("Too many IMU errors. Reinitializing BNO055...\n");
    bno055_set_operation_mode(BNO055_OPERATION_MODE_CONFIG);
    usleep(50000);
    
    bno055_set_sys_rst(BNO055_BIT_ENABLE);
    usleep(800000); 
    bno055_init(&bno); 
    
    bno055_set_power_mode(BNO055_POWER_MODE_NORMAL);
    bno055_set_operation_mode(BNO055_OPERATION_MODE_CONFIG);
    usleep(50000);
    
    bno055_set_axis_remap_value(BNO055_DEFAULT_AXIS); 
    bno055_set_remap_x_sign(BNO055_REMAP_AXIS_POSITIVE);
    bno055_set_remap_y_sign(BNO055_REMAP_AXIS_POSITIVE);
    bno055_set_remap_z_sign(BNO055_REMAP_AXIS_POSITIVE);
    
    bno055_set_operation_mode(BNO055_OPERATION_MODE_IMUPLUS);
    usleep(50000);
    printf("BNO055 Reinitialized.\n");
}

void *imu_thread_func(void *arg) {
    printf("IMU background thread started.\n");
    int thread_imu_error_count = 0;
    while (keep_running) {
        struct bno055_quaternion_t quat = {0, 0, 0, 0};
        s8 res = read_quaternion_sandwich(&quat);
        
        if (res != BNO055_SUCCESS) {
            thread_imu_error_count++;
            printf("I2C Read Error in background! Consecutive errors: %d\n", thread_imu_error_count);
            if (thread_imu_error_count >= 5) {
                reinit_imu();
                thread_imu_error_count = 0;
            }
        } else {
            if (quat.w == 0 && quat.x == 0 && quat.y == 0 && quat.z == 0) {
                thread_imu_error_count++;
                printf("IMU returned all zeros in background! Consecutive errors: %d\n", thread_imu_error_count);
                if (thread_imu_error_count >= 5) {
                    reinit_imu();
                    thread_imu_error_count = 0;
                }
            } else {
                thread_imu_error_count = 0;
            }
        }

        float qw = quat.w / 16384.0f;
        float qx = quat.x / 16384.0f;
        float qy = quat.y / 16384.0f;
        float qz = quat.z / 16384.0f;

        u8 calib_stat = 0;
        s8 r_cal = bno055_i2c_bus_read(BNO055_ADDRESS_A, BNO055_CALIB_STAT_ADDR, &calib_stat, 1);
        u8 sys_cal   = (calib_stat >> 6) & 0x03;
        u8 gyro_cal  = (calib_stat >> 4) & 0x03;
        u8 accel_cal = (calib_stat >> 2) & 0x03;
        u8 mag_cal   = calib_stat & 0x03;

        pthread_mutex_lock(&imu_lock);
        shared_imu_data.qw = qw;
        shared_imu_data.qx = qx;
        shared_imu_data.qy = qy;
        shared_imu_data.qz = qz;
        if (r_cal == BNO055_SUCCESS) {
            shared_imu_data.sys_cal   = sys_cal;
            shared_imu_data.gyro_cal  = gyro_cal;
            shared_imu_data.accel_cal = accel_cal;
            shared_imu_data.mag_cal   = mag_cal;
        }
        pthread_mutex_unlock(&imu_lock);

        // BNO055 updates internal fusion data at 100 Hz.
        // A single sandwich read + cal read takes ~7 ms on the 400kHz I2C bus.
        // Sleeping for 3 ms yields a ~10 ms cycle (100 Hz).
        usleep(3000);
    }
    return NULL;
}

void *processing_thread(void *arg) {
    int read_idx = 0;
    int32_t *sum_pulse = (int32_t *)malloc(pulse_len * sizeof(int32_t));
    
    // send_buf format: [num_pulses, qw, qx, qy, qz, sys_cal, gyro_cal, accel_cal, mag_cal, samples...]
    float *send_buf = (float *)malloc((pulse_len + 9) * sizeof(float));

    double total_wait = 0;
    double total_proc = 0;
    double total_imu = 0;
    double total_send = 0;
    int profile_count = 0;

    printf("Processing & Network thread started.\n");

    while (keep_running) {
        double t_start = get_time_us();
        
        int32_t * restrict sum_ptr = sum_pulse;
        memset(sum_ptr, 0, pulse_len * sizeof(int32_t));
        int num_pulses = 0;

        double t_after_wait = t_start;
        double t_after_proc = t_start;

        for (int b = 0; b < buffers_to_average; b++) {
            pthread_mutex_lock(&lock);
            while (!buffer_ready[read_idx] && keep_running) {
                pthread_cond_wait(&cond, &lock);
            }
            if (!keep_running) {
                pthread_mutex_unlock(&lock);
                break;
            }
            pthread_mutex_unlock(&lock);

            if (b == 0) {
                t_after_wait = get_time_us();
                total_wait += (t_after_wait - t_start);
            }

            const int16_t * restrict buf = buffers[read_idx];
            int i = 1000;
            int coarse_step = 25;
            
            while (i < TOTAL_SAMPLES - pulse_len && num_pulses < 1000 * buffers_to_average) {
                if (abs(buf[i]) > threshold_counts) {
                    // Fine Search: backtrace to find the exact first sample crossing the threshold
                    int exact_i = i;
                    for (int k = i - 1; k > i - coarse_step && k >= 1000; k--) {
                        if (abs(buf[k]) > threshold_counts) {
                            exact_i = k;
                        }
                    }
                    
                    int start_idx = exact_i - PRE_TRIGGER;
                    if (start_idx < 0) start_idx = 0;
                    const int16_t * restrict pulse_src = &buf[start_idx];
                    for (int j = 0; j < pulse_len; j++) {
                        sum_ptr[j] += pulse_src[j];
                    }
                    num_pulses++;
                    i = exact_i + holdoff_samples;
                } else {
                    i += coarse_step;
                }
            }

            pthread_mutex_lock(&lock);
            buffer_ready[read_idx] = false;
            pthread_mutex_unlock(&lock);
            read_idx = (read_idx + 1) % NUM_BUFFERS;
        }

        if (!keep_running) break;

        t_after_proc = get_time_us();
        total_proc += (t_after_proc - t_after_wait);

        // 1. Read IMU Quaternions from shared state
        float qw, qx, qy, qz;
        uint8_t sys_cal, gyro_cal, accel_cal, mag_cal;
        pthread_mutex_lock(&imu_lock);
        qw = shared_imu_data.qw;
        qx = shared_imu_data.qx;
        qy = shared_imu_data.qy;
        qz = shared_imu_data.qz;
        sys_cal = shared_imu_data.sys_cal;
        gyro_cal = shared_imu_data.gyro_cal;
        accel_cal = shared_imu_data.accel_cal;
        mag_cal = shared_imu_data.mag_cal;
        pthread_mutex_unlock(&imu_lock);

        double t_after_imu = get_time_us();
        total_imu += (t_after_imu - t_after_proc);

        // 2. Package Packet
        float * restrict send_ptr = send_buf;
        send_ptr[0] = (float)num_pulses;
        send_ptr[1] = qw;
        send_ptr[2] = qx;
        send_ptr[3] = qy;
        send_ptr[4] = qz;
        send_ptr[5] = (float)sys_cal;
        send_ptr[6] = (float)gyro_cal;
        send_ptr[7] = (float)accel_cal;
        send_ptr[8] = (float)mag_cal;

        if (num_pulses > 0) {
            float mean_val = 0.0f;
            for (int j = 0; j < pulse_len; j++) {
                send_ptr[j + 9] = ((float)sum_ptr[j] / num_pulses) * (20.0 / 8192.0);
                mean_val += send_ptr[j + 9];
            }
            mean_val /= pulse_len;
            for (int j = 0; j < pulse_len; j++) {
                send_ptr[j + 9] -= mean_val;
            }
        } else {
            for (int j = 0; j < pulse_len; j++) send_ptr[j + 9] = 0.0f;
        }

        double t_start_send = get_time_us();

        // 3. Send over network
        int total_bytes = (pulse_len + 9) * sizeof(float);
        int sent_bytes = 0;
        while (sent_bytes < total_bytes) {
            int n = send(sock, (char*)send_buf + sent_bytes, total_bytes - sent_bytes, MSG_NOSIGNAL);
            if (n <= 0) {
                printf("Connection lost. Attempting to reconnect...\n");
                close(sock);
                
                // User requested to re-initialize the BNO055 on every restart
                reinit_imu();
                
                while (keep_running) {
                    sock = socket(AF_INET, SOCK_STREAM, 0);
                    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) == 0) {
                        printf("Reconnected!\n");
                        break;
                    }
                    close(sock);
                    usleep(1000000);
                }
                break;
            }
            sent_bytes += n;
        }

        double t_after_send = get_time_us();
        total_send += (t_after_send - t_start_send);

        profile_count++;
        if (profile_count >= 100) {
            printf("[PROFILE] Frames: 100 | Wait: %.2f ms | PulseProc: %.2f ms | IMU: %.2f ms | Send: %.2f ms\n",
                   (total_wait / profile_count) / 1000.0,
                   (total_proc / profile_count) / 1000.0,
                   (total_imu / profile_count) / 1000.0,
                   (total_send / profile_count) / 1000.0);
            total_wait = 0;
            total_proc = 0;
            total_imu = 0;
            total_send = 0;
            profile_count = 0;
        }

    }
    
    free(sum_pulse);
    free(send_buf);
    return NULL;
}

int main(int argc, char **argv) {
    signal(SIGINT, handle_sigint);
    signal(SIGTERM, handle_sigint);
    char *server_ip = "169.254.240.193"; // Updated to current Mac IP
    if (argc > 1) server_ip = argv[1];
    if (argc > 2) threshold_counts = atoi(argv[2]);
    if (argc > 3) {
        pulse_len = atoi(argv[3]);
        holdoff_samples = pulse_len;
    }
    if (argc > 4) {
        buffers_to_average = atoi(argv[4]);
        if (buffers_to_average < 1) buffers_to_average = 1;
    }

    // AXI INIT
    if (rp_Init() != RP_OK) return -1;
    rp_CalibrationReset(false, false);

    uint32_t axi_start, axi_size;
    rp_AcqAxiGetMemoryRegion(&axi_start, &axi_size);

    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    void *mmap_ptr = mmap(NULL, axi_size, PROT_READ, MAP_SHARED, mem_fd, axi_start);

    rp_AcqReset();
    rp_AcqSetDecimation(RP_DEC_1);
    rp_AcqAxiSetDecimationFactor(RP_DEC_1);
    rp_AcqAxiSetTriggerDelay(RP_CH_1, TOTAL_SAMPLES);
    rp_AcqAxiSetBufferSamples(RP_CH_1, axi_start, TOTAL_SAMPLES);
    rp_AcqAxiEnable(RP_CH_1, true);

    // I2C INIT (BNO055)
    if ((i2c_fd = open("/dev/i2c-0", O_RDWR)) < 0) {
        perror("Failed to open I2C bus 0");
        return -1;
    }
    if (ioctl(i2c_fd, I2C_SLAVE, BNO055_ADDRESS_A) < 0) {
        perror("Failed to find BNO055");
        return -1;
    }
    
    bno.bus_write = bno055_i2c_bus_write;
    bno.bus_read = bno055_i2c_bus_read;
    bno.delay_msec = bno055_delay_msek;
    bno.dev_addr = BNO055_ADDRESS_A;

    if (bno055_init(&bno) < 0) {
        printf("BNO055 Init Failed!\n");
        return -1;
    }
    
    // Reset and wait
    bno055_set_sys_rst(BNO055_BIT_ENABLE);
    usleep(800000); 
    bno055_init(&bno); 
    
    // Config Mode
    bno055_set_power_mode(BNO055_POWER_MODE_NORMAL);
    bno055_set_operation_mode(BNO055_OPERATION_MODE_CONFIG);
    usleep(50000);
    
    // Remap axes (Picoscope matching: 0,1,2,0,1,1)
    bno055_set_axis_remap_value(BNO055_DEFAULT_AXIS); // 0x24
    bno055_set_remap_x_sign(BNO055_REMAP_AXIS_POSITIVE);
    bno055_set_remap_y_sign(BNO055_REMAP_AXIS_POSITIVE);
    bno055_set_remap_z_sign(BNO055_REMAP_AXIS_POSITIVE);
    
    // IMU Mode (Gyro + Accel only, no Magnetometer)
    bno055_set_operation_mode(BNO055_OPERATION_MODE_IMUPLUS);
    usleep(50000);
    printf("BNO055 Initialized in IMUPLUS mode using official API.\n");

    // SOCKET INIT
    sock = socket(AF_INET, SOCK_STREAM, 0);
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
    inet_pton(AF_INET, server_ip, &serv_addr.sin_addr);
    
    printf("Connecting to Host at %s:%d...\n", server_ip, PORT);
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        printf("Connection Failed!\n");
        return -1;
    }

    for (int i = 0; i < NUM_BUFFERS; i++) buffers[i] = (int16_t *)malloc(TOTAL_SAMPLES * sizeof(int16_t));

    pthread_t p_thread;
    pthread_create(&p_thread, NULL, processing_thread, NULL);

    pthread_t imu_thread;
    pthread_create(&imu_thread, NULL, imu_thread_func, NULL);

    printf("Starting AXI + IMU Integrated Stream (averaging %d buffers of 8ms per frame)...\n", buffers_to_average);
    int write_idx = 0;
    while (keep_running) {
        bool can_write = false;
        while (!can_write && keep_running) {
            pthread_mutex_lock(&lock);
            if (!buffer_ready[write_idx]) can_write = true;
            pthread_mutex_unlock(&lock);
            if (!can_write) usleep(100);
        }
        if (!keep_running) break;

        rp_AcqStart();
        rp_AcqSetTriggerSrc(RP_TRIG_SRC_NOW);
        bool fill_state = false;
        while (!fill_state && keep_running) rp_AcqAxiGetBufferFillState(RP_CH_1, &fill_state);
        rp_AcqStop();

        memcpy(buffers[write_idx], mmap_ptr, TOTAL_SAMPLES * sizeof(int16_t));
        
        pthread_mutex_lock(&lock);
        buffer_ready[write_idx] = true;
        pthread_cond_signal(&cond);
        pthread_mutex_unlock(&lock);

        write_idx = (write_idx + 1) % NUM_BUFFERS;
    }

    keep_running = false;
    pthread_cond_signal(&cond);
    pthread_join(p_thread, NULL);
    pthread_join(imu_thread, NULL);
    for (int i = 0; i < NUM_BUFFERS; i++) free(buffers[i]);
    close(sock);
    close(i2c_fd);
    rp_AcqAxiEnable(RP_CH_1, false);
    rp_Release();
    return 0;
}
