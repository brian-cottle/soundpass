#!/usr/bin/expect -f
set timeout 10
spawn scp -o StrictHostKeyChecking=no stream_axi_imu.c root@169.254.135.224:~/
expect {
    "password:" {
        send "root\r"
        exp_continue
    }
    eof
}
spawn ssh -o StrictHostKeyChecking=no root@169.254.135.224 "gcc -O3 -march=armv7-a -mtune=cortex-a9 -mfpu=neon -mfloat-abi=hard -I/opt/redpitaya/include -L/opt/redpitaya/lib -Wl,-rpath,/opt/redpitaya/lib stream_axi_imu.c bno055.c -o stream_axi_imu -lrp -lrp-hw-calib -lpthread -lm"
expect {
    "password:" {
        send "root\r"
        exp_continue
    }
    eof
}
