import numpy as np

from sklearn.metrics import confusion_matrix


                                                          


                                                                   


def evaluate_model_criteria(y_true, y_pred, positive_label=1, sample_device_mapping=None):

    """
    Evaluates model performance based on true positive and false positive rates.
    Extends the evaluation to include device IDs for misclassified samples.

    Args:
    - y_true (np.ndarray): True labels.
    - y_pred (np.ndarray): Predicted labels.
    - positive_label (int): The label considered as positive class (default is 1).
    - sample_device_mapping (list): Optional. List mapping each sample index to its device ID.

    Returns:
    - Tuple[float, float, list]: True Detection Rate (TDR), False Detection Rate (FDR), 
      and a list of misclassified sample indices or tuples of (index, device ID) if sample_device_mapping is provided.
    """

                              

    cm = confusion_matrix(y_true, y_pred, labels=[positive_label, 1-positive_label])

    

                                                                        

    TDR = cm[0, 0] / np.sum(cm[0, :]) if np.sum(cm[0, :]) > 0 else 0

    FDR = cm[1, 0] / np.sum(cm[1, :]) if np.sum(cm[1, :]) > 0 else 0

    

                                    

    misclassified_indices = np.where(y_true != y_pred)[0]

    

                                                                                       

    if sample_device_mapping is not None:

        misclassified_info = [(index, sample_device_mapping[index]) for index in misclassified_indices]

    else:

        misclassified_info = misclassified_indices.tolist()

    

    return TDR, FDR, misclassified_info


def calculate_closeness_score(auth_TDR, auth_FDR, rogue_TDR, rogue_FDR, TDR_threshold=0.95, FDR_threshold=0.05):

    """
    Calculates a score representing how close a model's performance is to the desired thresholds.

    Args:
    - auth_TDR (float): True Detection Rate for authenticated devices.
    - auth_FDR (float): False Detection Rate for authenticated devices.
    - rogue_TDR (float): True Detection Rate for rogue devices.
    - rogue_FDR (float): False Detection Rate for rogue devices.
    - TDR_threshold (float): Desired true positive rate threshold (default is 0.95).
    - FDR_threshold (float): Desired false positive rate threshold (default is 0.05).

    Returns:
    - float: Closeness score representing the model's performance relative to the desired thresholds.
    """

                                   

    score = (

        abs(auth_TDR - TDR_threshold) + 

        abs(auth_FDR - FDR_threshold) + 

        abs(rogue_TDR - TDR_threshold) + 

        abs(rogue_FDR - FDR_threshold)

    )

    return score
