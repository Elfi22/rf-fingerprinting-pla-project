import numpy as np

from scipy.signal import butter, lfilter, cheby1


                                                            


                                                                      


def load_iq_data(filepath, start_index, end_index):

    """
    Load IQ data from a binary file.

    Args:
    - filepath (str): Path to the binary file containing IQ data.
    - start_index (int): Starting index for data extraction.
    - end_index (int): Ending index for data extraction.

    Returns:
    - np.ndarray: Loaded IQ data as a complex64 numpy array.
    """

    with open(filepath, 'rb') as f:

        f.seek(start_index * np.dtype(np.complex64).itemsize, 0)

        iq_data = np.fromfile(f, dtype=np.complex64, count=end_index - start_index)

    return iq_data


def normalize_magnitude(iq_data):

    """
    Normalize the magnitude of IQ data.

    Args:
    - iq_data (np.ndarray): Input IQ data.

    Returns:
    - np.ndarray: Normalized magnitude of the IQ data.
    """

    return np.abs(iq_data) / np.max(np.abs(iq_data))


                                                                       


def detect_transients(normalized_magnitude, sampling_rate, transient_threshold, min_transient_duration):

    """
    Detect transient signals based on a magnitude threshold.

    Args:
    - normalized_magnitude (np.ndarray): Normalized magnitude of the signal.
    - sampling_rate (float): Sampling rate of the signal.
    - transient_threshold (float): Threshold for detecting transients.
    - min_transient_duration (float): Minimum duration for a transient signal to be considered.

    Returns:
    - tuple: Start and end indices of detected transients.
    """

    transient_indices = np.where(normalized_magnitude > transient_threshold)[0]

    min_transient_samples = int(min_transient_duration * sampling_rate)

    transient_boundaries = np.diff(transient_indices) > min_transient_samples

    transient_start_indices = transient_indices[np.insert(transient_boundaries, 0, True)]

    transient_end_indices = transient_indices[np.append(transient_boundaries, True)]

    

                                                                                    

    if len(transient_start_indices) > 1:                                            

        transient_start_indices = transient_start_indices[1:]                                  

        transient_end_indices = transient_end_indices[1:]                                

    return transient_start_indices, transient_end_indices


def filter_transients(transient_start_indices, transient_end_indices, normalized_magnitude, sampling_rate, specific_duration_threshold, specific_magnitude_threshold):

    """
    Filter transients based on specific duration and magnitude thresholds.

    Args:
    - transient_start_indices (np.ndarray): Start indices of detected transients.
    - transient_end_indices (np.ndarray): End indices of detected transients.
    - normalized_magnitude (np.ndarray): Normalized magnitude of the signal.
    - sampling_rate (float): Sampling rate of the signal.
    - specific_duration_threshold (float): Minimum duration threshold for selecting transients.
    - specific_magnitude_threshold (float): Minimum magnitude threshold for selecting transients.

    Returns:
    - list: List of tuples containing start and end indices of selected transients.
    """

    selected_transients = []

    for start, end in zip(transient_start_indices, transient_end_indices):

        if end > start:                                 

            duration = (end - start) / sampling_rate

            max_magnitude = np.max(normalized_magnitude[start:end])

            if duration > specific_duration_threshold and max_magnitude > specific_magnitude_threshold:

                selected_transients.append((start, end))

    return selected_transients


def apply_lowpass_filter(data, cutoff_freq, sample_rate, filter_type='chebyshev', order=4, ripple=0.5):

    """
    Apply a low-pass filter to the data based on the specified filter type.

    Args:
    - data (np.ndarray): The input signal.
    - cutoff_freq (float): The cutoff frequency of the filter in Hz.
    - sample_rate (float): The sampling rate of the signal in Hz.
    - filter_type (str): The type of filter to apply. Options: 'chebyshev', 'butter'.
    - order (int): The order of the filter (default is 6 for Butterworth).
    - ripple (float): The maximum ripple allowed in the passband (in dB, default is 0.5 for Chebyshev).

    Returns:
    - np.ndarray: The filtered signal.
    """

    nyquist_freq = 0.5 * sample_rate

    normal_cutoff = cutoff_freq / nyquist_freq


    if filter_type == 'chebyshev':

                                           

        b, a = cheby1(order, ripple, normal_cutoff, btype='low', analog=False)

    elif filter_type == 'butter':

                                             

        b, a = butter(order, normal_cutoff, btype='low', analog=False)

    else:

        raise ValueError(f"Unknown filter type: {filter_type}. Available types: 'chebyshev', 'butter'.")


                                  

    filtered_data = lfilter(b, a, data)

    return filtered_data
