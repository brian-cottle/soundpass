#!/bin/bash

# Define the python path from the conda environment directly
PYTHON_EXEC="/opt/anaconda3/envs/soundpass/bin/python"

# Trap Ctrl+C (SIGINT) and exit to kill child processes
trap 'echo "Stopping SoundPass..."; kill $(jobs -p) 2>/dev/null; exit 0' SIGINT SIGTERM EXIT

echo "Starting SoundPass Live GUI..."
# Start the Python GUI in the background
$PYTHON_EXEC soundpass_live_gui.py &

# Red Pitaya Data Stream Configuration
THRESHOLD=100
PULSE_LEN=6000
AVERAGING=1

echo "Connecting to Red Pitaya to start data stream..."
# Create a temporary expect script to handle the SSH login
EXPECT_SCRIPT=$(mktemp)
cat << EXPECT_EOF > $EXPECT_SCRIPT
set timeout -1
spawn ssh -o StrictHostKeyChecking=no root@169.254.135.224 "MAC_IP=\\\$(echo \\\$SSH_CLIENT | awk '{print \\\$1}'); echo 'Auto-detected Mac IP: '\\\$MAC_IP; ./stream_axi_imu \\\$MAC_IP $THRESHOLD $PULSE_LEN $AVERAGING"

expect {
    "password:" {
        send "root\r"
        exp_continue
    }
    eof
}
EXPECT_EOF

# Run the expect script in the foreground so Ctrl+C breaks out of it
expect -f $EXPECT_SCRIPT

# Clean up temp script
rm -f $EXPECT_SCRIPT

# Wait for the python GUI to close before exiting the script entirely
wait
