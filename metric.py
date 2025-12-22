import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import accuracy_score


def calculate_discriminative_tv_qda(real_data, generated_data, n_splits=10):
    """
    Estimates the Total Variation (TV) Distance by averaging the result over
    multiple random train/test splits (repeated random sub-sampling).
    
    Args:
        real_data (np.ndarray): Samples from the real distribution (P).
        generated_data (np.ndarray): Samples from the generated distribution (Q).
        n_splits (int): The number of random splits to average over for stabilization.
            
    Returns:
        float: Stabilized estimated Total Variation Distance (TVD).
    """
    
    # 1. Prepare Data for Classification (Done once)
    X = np.concatenate([real_data, generated_data])
    y = np.concatenate([np.ones(len(real_data)), np.zeros(len(generated_data))])
    
    # List to store TV estimates from each split
    tv_estimates = []
    
    # Iterate through the desired number of splits
    for i in range(n_splits):
        # The key difference: random_state=i ensures a different split each time
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=i, stratify=y
        )
        
        try:
            # 2. Train the Bayes Optimal Classifier (QDA)
            classifier = QDA()
            classifier.fit(X_train, y_train)
            
            # 3. Calculate the Classification Error (Risk, R_hat) on the test set
            accuracy = classifier.score(X_test, y_test)
            risk_R_hat = 1.0 - accuracy
            
            # 4. Estimate TV Distance: TV(P, Q) = 1 - 2 * R_hat
            estimated_tv_distance = 1.0 - 2 * risk_R_hat
            
            # Clamp and store the result
            tv_estimates.append(max(0.0, min(1.0, estimated_tv_distance)))
            
        except ValueError as e:
            # Handle potential QDA errors (e.g., singular matrix due to small samples)
            # print(f"Error training classifier on split {i}: {e}")
            continue # Skip this split
            
    if not tv_estimates:
        return np.nan # Return NaN if no successful splits occurred
        
    # 5. Return the mean of all successful TV estimates
    return np.mean(tv_estimates)


def kl_divergence_gaussians(mean_p, cov_p, mean_q, cov_q):
    """
    Calculate the KL divergence D_KL(P || Q) between two multivariate Gaussian distributions P and Q.
    
    Args:
        mean_p (np.ndarray): Mean vector of distribution P.
        cov_p (np.ndarray): Covariance matrix of distribution P.
        mean_q (np.ndarray): Mean vector of distribution Q.
        cov_q (np.ndarray): Covariance matrix of distribution Q.        
    """
    d = mean_p.shape[0]
    cov_q_inv = np.linalg.inv(cov_q)
    
    term1 = np.trace(cov_q_inv @ cov_p)
    term2 = (mean_q - mean_p).T @ cov_q_inv @ (mean_q - mean_p)
    sign_q, abslogdet_q = np.linalg.slogdet(cov_q)
    sign_p, abslogdet_p = np.linalg.slogdet(cov_p)
    term3 = sign_q * abslogdet_q - sign_p * abslogdet_p
    #term3 = np.log(np.linalg.det(cov_q) / np.linalg.det(cov_p) + 1e-10)  # Added small constant for numerical stability
    
    kl_div = 0.5 * (term1 + term2 - d + term3)
    return kl_div
