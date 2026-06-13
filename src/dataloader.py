import os

import numpy as np

import re

import h5py

from src.preprocessing import (

    load_iq_data, normalize_magnitude, detect_transients,

    filter_transients, apply_lowpass_filter

)

from src.features_generation import generate_rf_dna_fingerprint


                                                    


def natural_sort_key(s):

    """
    Extracts all numbers from a string and converts them to integers for natural sorting.

    Args:
    - s (str): The input string.

    Returns:
    - list: List of strings and integers for natural sorting.
    """

    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def load_data_and_unique_labels(

        data_directory, sample_rate=20e6, cutoff_freq=1.5e6, M=150, KG=150, N=1,

        NP=50, NT=15, NF=15, transient_threshold=0.6, specific_duration_threshold=0.003,

        specific_magnitude_threshold=0.4, min_transient_duration=0.005,

        filter_type='chebyshev', filter_order=4, filter_ripple=0.5, mode='diagonal', save_path='processed_data.h5'):

    """
    Loads IQ data and generates RF fingerprints, ensuring unique labels for each device.

    Args:
    - data_directory (str): Path to the directory containing device data folders.
    - sample_rate (float): The sampling rate of the signal in Hz.
    - cutoff_freq (float): The cutoff frequency of the low-pass filter in Hz.
    - M, KG, N, NP, NT, NF (int): Parameters for RF DNA fingerprint generation.
    - transient_threshold (float): Threshold for transient detection.
    - specific_duration_threshold (float): Specific duration threshold for filtering transients.
    - specific_magnitude_threshold (float): Specific magnitude threshold for filtering transients.
    - min_transient_duration (float): Minimum duration for a detected transient.
    - filter_type (str): The type of low-pass filter to apply. Options: 'chebyshev', 'butter'.
    - filter_order (int): The order of the low-pass filter.
    - filter_ripple (float): The maximum ripple allowed in the passband (in dB), applicable for Chebyshev filter.
    - save_path (str): Path to save the processed data in HDF5 format.

    Returns:
    - Tuple[np.ndarray, np.ndarray, dict, list]: The fingerprints data array, labels array, device ID mapping dictionary, and list of total devices.
    """

                                                  

    data = []

    labels = []


                                                                      

    device_folders = [folder for folder in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, folder))]

    device_folders.sort(key=natural_sort_key)                                                               

    device_id_mapping = {folder: idx for idx, folder in enumerate(device_folders)}

    total_devices = list(device_id_mapping.keys())                                  


    for device_name, device_id in device_id_mapping.items():

        print(device_name)

        folder_path = os.path.join(data_directory, device_name)

                            

        for file in sorted(os.listdir(folder_path), key=natural_sort_key):

                         

            if file.endswith('.bin'):

                filepath = os.path.join(folder_path, file)

                try:

                                                

                    iq_data = load_iq_data(filepath, 0, -1)

                                                                      

                                                                         

                    

                                                            

                    transient_start_indices, transient_end_indices = detect_transients(

                        iq_data, sample_rate, transient_threshold, min_transient_duration

                    )

                    

                                                                                                    

                    selected_transients = filter_transients(

                        transient_start_indices, transient_end_indices, iq_data, sample_rate,

                        specific_duration_threshold, specific_magnitude_threshold

                    )


                    for start, end in selected_transients:

                        if end > start:                                 

                            transient_segment = iq_data[start:end]

                                                                                          

                            filtered_transient = apply_lowpass_filter(

                                transient_segment, cutoff_freq, sample_rate,

                                filter_type=filter_type, order=filter_order, ripple=filter_ripple

                            )

                                                                                     

                            fingerprint = generate_rf_dna_fingerprint(

                                filtered_transient, fs=sample_rate, M=M, KG=KG, N=N, NP=NP, NT=NT, NF=NF, mode=mode

                            )

                            print(fingerprint)

                                                                                                                       

                            data.append(fingerprint)

                            labels.append(device_id)

                except ValueError as e:

                    print(f"Skipping file {file} due to error: {e}")

    

    data = np.array(data)

    labels = np.array(labels)


                                      

    with h5py.File(save_path, 'w') as f:

        f.create_dataset('data', data=data)

        f.create_dataset('labels', data=labels)

        f.create_dataset('device_id_mapping', data=np.string_(str(device_id_mapping)))

        f.create_dataset('total_devices', data=np.string_(str(total_devices)))


    return data, labels, device_id_mapping, total_devices


                                       

def load_data_from_hdf5(file_path):

    """
    Loads processed data from an HDF5 file.

    Args:
    - file_path (str): Path to the HDF5 file.

    Returns:
    - Tuple[np.ndarray, np.ndarray, dict, list]: The fingerprints data array, labels array, device ID mapping dictionary, and list of total devices.
    """

    with h5py.File(file_path, 'r') as f:

        data = f['data'][:]

        labels = f['labels'][:]

        device_id_mapping = eval(f['device_id_mapping'][()].decode())

        total_devices = eval(f['total_devices'][()].decode())


    return data, labels, device_id_mapping, total_devices
