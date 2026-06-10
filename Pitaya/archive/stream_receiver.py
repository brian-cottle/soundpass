import socket
import numpy as np
import os

PORT = 5005
TOTAL_SAMPLES = 1000000
BUFFER_SIZE = TOTAL_SAMPLES * 2 # 2 bytes per int16 (int16_t)

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Allow address reuse
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', PORT))
    server_socket.listen(1)
    print(f"Listening on port {PORT}...")
    print("Waiting for Red Pitaya to connect...")

    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    frame_count = 0
    try:
        while True:
            # Receive exactly BUFFER_SIZE bytes
            data = bytearray()
            while len(data) < BUFFER_SIZE:
                packet = conn.recv(BUFFER_SIZE - len(data))
                if not packet:
                    print("Connection lost.")
                    return
                data.extend(packet)
            
            # Convert to numpy array
            big_buffer = np.frombuffer(data, dtype=np.int16)
            
            # Atomic save: Write to temp file then rename
            # This prevents Streamlit or other processes from reading a partial file
            with open('data_raw.bin.tmp', 'wb') as f:
                big_buffer.tofile(f)
            os.replace('data_raw.bin.tmp', 'data_raw.bin')
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Received and saved {frame_count} frames...")
                
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        conn.close()
        server_socket.close()

if __name__ == "__main__":
    start_server()
