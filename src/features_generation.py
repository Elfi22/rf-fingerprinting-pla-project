import numpy as np

from scipy.stats import skew, kurtosis


                                              


                                                                          


def gaussian_window(length, std_dev):

    """
    Generates a Gaussian window.

    Args:
    - length (int): The length of the window.
    - std_dev (float): The standard deviation of the Gaussian function.

    Returns:
    - np.ndarray: The generated Gaussian window.
    """

    n = np.arange(length)

    return np.exp(-0.5 * ((n - length // 2) / std_dev) ** 2)


def dgt(signal, M=150, KG=150, N=1):

    """
    Computes the Discrete Gabor Transform (DGT) of a signal.

    Args:
    - signal (np.ndarray): The input signal.
    - M (int): The number of time frames.
    - KG (int): The number of frequency bins.
    - N (int): The oversampling factor.

    Returns:
    - np.ndarray: The DGT of the signal.
    """

    MN = M * N

    length = len(signal)

    Gmk = np.zeros((M, KG), dtype=np.complex_)

    window = gaussian_window(MN, std_dev=MN // 8)                   


    for m in range(M):

        for k in range(KG):

            for n in range(MN):

                index = n + m * N

                if index < length:                                        

                    exp_term = np.exp(-1j * 2 * np.pi * k * n / KG)

                    windowed_signal = signal[index] * window[n] * exp_term

                    Gmk[m, k] += windowed_signal

                else:

                    break                                                    


    return Gmk


def normalize_gabor_coefficients(Gmk):

    """
    Normalizes Gabor coefficients.

    Args:
    - Gmk (np.ndarray): The input Gabor coefficients.

    Returns:
    - np.ndarray: The normalized Gabor coefficients.
    """

    return (Gmk - np.min(Gmk)) / (np.max(Gmk) - np.min(Gmk))


def entropy(signal):

    """
    Calculates the Shannon entropy of a signal.

    Args:
    - signal (np.ndarray): Input signal vector.

    Returns:
    - float: The entropy of the signal.
    """

                                                         

    hist, bin_edges = np.histogram(signal, bins='auto', density=True)

    prob_distribution = hist * np.diff(bin_edges)


                                                   

    prob_distribution = prob_distribution[prob_distribution > 0]


                       

    return -np.sum(prob_distribution * np.log2(prob_distribution))


def extract_features_from_patches(Zxx_normalized, NP=50, NT=15, NF=10, mode='diagonal'):

    """
    Extracts features from patches of normalized Gabor coefficients.

    Args:
    - Zxx_normalized (np.ndarray): The normalized Gabor coefficients.
    - NP (int): Number of patches.
    - NT (int): Time dimension of each patch.
    - NF (int): Frequency dimension of each patch.
    - mode (str): The mode of patch extraction ('horizontal', 'vertical', or 'diagonal').

    Returns:
    - np.ndarray: The extracted features.
    """

    patches_features = []

    rows, cols = Zxx_normalized.shape


    def get_patch_start_positions(mode):

        """
        Determines the start positions of patches based on the mode.

        Args:
        - mode (str): The mode of patch extraction ('horizontal', 'vertical', or 'diagonal').

        Returns:
        - list: A list of tuples containing the start positions (row, col) of each patch.
        """

        positions = []

        if mode == 'horizontal':

            num_patches_per_row = np.ceil(cols / NF).astype(int)

            for pt in range(NP):

                row_index = pt // num_patches_per_row

                col_index = pt % num_patches_per_row

                positions.append((row_index * NT, col_index * NF))

        elif mode == 'vertical':

            num_patches_per_col = np.ceil(rows / NT).astype(int)

            for pt in range(NP):

                col_index = pt // num_patches_per_col

                row_index = pt % num_patches_per_col

                positions.append((row_index * NT, col_index * NF))

        elif mode == 'diagonal':

            for i in range(min(cols // NF, rows // NT)):

                if len(positions) < NP:

                    positions.append((i * NT, i * NF))

            if len(positions) < NP:

                for i in range(min(cols // NF, rows // NT)):

                    if len(positions) < NP:

                        positions.append((i * NT, (cols - (i + 1) * NF)))

            remaining_patches = NP - len(positions)

            if remaining_patches > 0:

                for row in range(0, rows, NT):

                    for col in range(0, cols, NF):

                        if (row, col) not in positions:

                            positions.append((row, col))

                            remaining_patches -= 1

                            if remaining_patches <= 0:

                                break

                    if remaining_patches <= 0:

                        break

        return positions


    patch_starts = get_patch_start_positions(mode)

    for start_row, start_col in patch_starts:

        end_row = min(start_row + NT, rows)

        end_col = min(start_col + NF, cols)

        if end_row > start_row and end_col > start_col:                         

            patch = Zxx_normalized[start_row:end_row, start_col:end_col]

            patch_vector = patch.reshape(-1)


                                            

            patch_features = [

                np.std(patch_vector),

                np.var(patch_vector),

                skew(patch_vector),

                kurtosis(patch_vector),

                entropy(patch_vector)

            ]

            patches_features.extend(patch_features)


                                                          

    entire_vector = Zxx_normalized.reshape(-1)

    entire_features = [

        np.std(entire_vector),

        np.var(entire_vector),

        skew(entire_vector),

        kurtosis(entire_vector),

        entropy(entire_vector)

    ]

    patches_features.extend(entire_features)


    return np.array(patches_features)


def generate_rf_dna_fingerprint(signal, fs=20e6, M=150, KG=150, N=1, NP=50, NT=15, NF=10, mode='diagonal'):

    """
    Generates RF-DNA fingerprints from a given signal, for each patch, based on DGT.

    Args:
    - signal (np.ndarray): The input RF signal.
    - fs (float): Sampling frequency of the signal.
    - M, KG, N (int): DGT calculation parameters.
    - NP (int): Number of patches to divide the TF representation into.
    - NT, NF (int): Dimensions of each patch.
    - mode (str): The mode of patch extraction ('horizontal', 'vertical', or 'diagonal').

    Returns:
    - np.ndarray: A vector of RF-DNA fingerprint features.
    """

                 

    Gmk = dgt(signal, M=M, KG=KG, N=N)


                                                  

    Zxx_normalized = normalize_gabor_coefficients(np.abs(Gmk) ** 2)


                                     

    fingerprint = extract_features_from_patches(Zxx_normalized, NP=NP, NT=NT, NF=NF, mode=mode)

    

    return fingerprint
