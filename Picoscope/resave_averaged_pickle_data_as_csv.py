# %%
# --- Imports ---
import os
import pickle
import numpy as np
import glob
from natsort import natsorted
from tqdm import tqdm
import matplotlib.pyplot as plt

# %%
# --- Configuration ---
# IMPORTANT: Set this to the directory containing your .pkl files.
# For example: DATA_DIR = "/path/to/your/dataset/1757955094_13"
DATA_DIR = '/Users/briancottle/Library/Mobile Documents/com~apple~CloudDocs/Consulting/SoundPass/data/artifact_examples/tissue_contact_no_double_wall/samples_3'

# %%
# --- File Scanning and Processing ---
print(f"Scanning for .pkl files in: {DATA_DIR}")
pickle_files = natsorted(glob.glob(os.path.join(DATA_DIR, "*.pkl")))

if not pickle_files:
    print("No .pkl files found in the specified directory.")
else:
    print(f"Found {len(pickle_files)} .pkl files. Starting processing...")

    # Loop through each file, process it, and save as .csv
    for idx, file_path in enumerate(tqdm(pickle_files, desc="Processing files")):
        try:
            # Load the data from the pickle file
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # The signal data is the first element in the loaded list
            all_samples = data[0]
            
            # Average the buffers along the first axis to get a single 1D signal
            averaged_signal = np.mean(all_samples, axis=0)
            
            # Define the output CSV file path
            # The new file will have the name of its index in the folder, e.g., '1.csv'
            output_filename = os.path.join(DATA_DIR, f"{idx + 1}.csv")
            
            # Save the averaged signal to the new CSV file
            np.savetxt(output_filename, averaged_signal, delimiter=",")
            
        except Exception as e:
            print(f"\nCould not process file {file_path}: {e}")

    print(f"\nProcessing complete. {len(pickle_files)} files were converted to .csv format.")


# %%
# --- Verification (Optional) ---
# This cell verifies that the data was saved correctly by loading and plotting one of the CSV files.

# Let's check the first generated CSV file
VERIFICATION_INDEX = 15
csv_to_check = os.path.join(DATA_DIR, f"{VERIFICATION_INDEX}.csv")

if os.path.exists(csv_to_check):
    print(f"Loading and plotting verification file: {csv_to_check}")
    
    # Load the data from the CSV
    loaded_signal = np.loadtxt(csv_to_check, delimiter=",")
    
    # Plot the signal
    plt.figure(figsize=(15, 5))
    plt.plot(loaded_signal)
    plt.title(f"Signal from {os.path.basename(csv_to_check)}")
    plt.xlabel("Sample Index")
    plt.ylabel("Averaged ADC Value")
    plt.grid(True)
    plt.show()
else:
    print(f"Verification file not found: {csv_to_check}. Please check if the processing step ran successfully.")
# %%