import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

class GaussianMixtureKNN:
    def __init__(self, n_clusters=3, n_samples=1000, n_features=2, mixture_probs=None, centers=None, random_state=42):
        """
        Initializes the Gaussian Mixture Model with KNN clustering.
        
        Parameters:
        - n_clusters: Number of Gaussian clusters
        - n_samples: Total number of samples
        - n_features: Number of features (dimensions)
        - mixture_probs: List of probabilities for each Gaussian component. Must sum to 1.
        - centers: Predefined cluster centers. If provided, overrides random generation of centers.
        - random_state: Seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.n_features = n_features
        self.mixture_probs = mixture_probs if mixture_probs else [1 / n_clusters] * n_clusters
        self.centers = centers
        self.random_state = random_state
        self.data, self.labels = None, None

        # Ensure mixture probabilities sum to 1
        if not np.isclose(sum(self.mixture_probs), 1):
            raise ValueError("The mixture probabilities must sum to 1.")

    def create_mixture(self):
        """
        Generates an N-mixture of Gaussian blobs based on specified mixture probabilities.
        
        Returns:
        - data: np.ndarray of generated data points
        - labels: np.ndarray of true cluster labels
        """
        cluster_samples = (np.array(self.mixture_probs) * self.n_samples).astype(int)
        
        self.data, self.labels = [], []
        for i, n_samples in enumerate(cluster_samples):
            center = self.centers[i] if self.centers is not None else None
            data, labels = make_blobs(n_samples=n_samples,
                                      centers=[center] if center is not None else 1,
                                      n_features=self.n_features,
                                      random_state=self.random_state + i)
            self.data.append(data)
            self.labels.extend([i] * n_samples)
        
        # Convert lists to numpy arrays
        self.data = np.vstack(self.data)
        self.labels = np.array(self.labels)
        
        print(f"Generated {self.n_clusters}-mixture of Gaussians with {self.n_samples} samples.")

    def perform_knn(self, n_neighbors=5):
        """
        Applies KNN clustering on the generated data.
        
        Parameters:
        - n_neighbors: Number of neighbors for the KNN algorithm
        
        Returns:
        - predicted_labels: Labels predicted by the KNN model
        """
        # Shuffle and split the data into training and testing sets
        np.random.seed(self.random_state)
        indices = np.random.permutation(len(self.data))
        train_size = int(0.8 * len(self.data))
        train_idx, test_idx = indices[:train_size], indices[train_size:]
        X_train, X_test = self.data[train_idx], self.data[test_idx]
        y_train, y_test = self.labels[train_idx], self.labels[test_idx]

        # Fit KNN model
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        
        # Predict on the test set
        predicted_labels = knn.predict(X_test)
        
        # Accuracy (optional)
        accuracy = accuracy_score(y_test, predicted_labels)
        print(f"KNN clustering accuracy: {accuracy * 100:.2f}%")
        
        # Concatenate back training and test labels for plotting
        self.predicted_labels = np.concatenate([y_train, predicted_labels])
        return self.predicted_labels

    def calculate_proportions(self):
        """
        Calculates and displays the proportion of each cluster.
        
        Returns:
        - cluster_proportions: Dictionary with cluster proportions
        """
        counts = Counter(self.predicted_labels)
        total_count = sum(counts.values())
        cluster_proportions = {label: count / total_count for label, count in counts.items()}
        print("Cluster Proportions:")
        for label, proportion in cluster_proportions.items():
            print(f"Cluster {label}: {proportion:.2%}")
        return cluster_proportions

    def plot_clusters(self):
        """
        Plots the clustered data with different colors for each cluster.
        """
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(self.data[:, 0], self.data[:, 1], c=self.predicted_labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ticks=range(self.n_clusters))
        plt.title(f"KNN Clustering of {self.n_clusters}-Mixture of Gaussians")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

# Usage example:
# gmm_knn = GaussianMixtureKNN(n_clusters=4, n_samples=1000, n_features=2, mixture_probs=[0.1, 0.3, 0.4, 0.2])
# gmm_knn.create_mixture()
# gmm_knn.perform_knn(n_neighbors=5)
# gmm_knn.calculate_proportions()
# gmm_knn.plot_clusters()




class LowrankGaussian:
    def __init__(self, n_samples=1000, n_features=10, rank=2, random_state=42):
        """
        Initializes the Low-Rank Gaussian distribution.
        
        Parameters:
        - n_samples: Number of samples to generate
        - n_features: Number of features (dimensions)
        - rank: Rank of the covariance matrix
        - random_state: Seed for reproducibility
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.rank = rank
        self.random_state = random_state
        self.data = None

    def generate_data(self):
        """
        Generates samples from a low-rank Gaussian distribution.
        
        Returns:
        - data: np.ndarray of generated data points
        """
        np.random.seed(self.random_state)
        
        # Generate a random mean vector
        mean = np.random.rand(self.n_features)
        
        # Generate a low-rank covariance matrix
        A = np.random.randn(self.n_features, self.rank)
        covariance = A @ A.T  # This ensures the covariance matrix is positive semi-definite
        
        # Generate samples
        self.data = np.random.multivariate_normal(mean, covariance, size=self.n_samples)
        
        print(f"Generated {self.n_samples} samples from a low-rank Gaussian distribution.")
        return self.data
    
    def plot_data(self):
        """
        Plots the generated data if it is 2D.
        """
        if self.n_features != 2:
            print("Plotting is only available for 2D data.")
            return
        
        plt.figure(figsize=(8, 6))
        plt.scatter(self.data[:, 0], self.data[:, 1], alpha=0.6)
        plt.title("Samples from Low-Rank Gaussian Distribution")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.axis('equal')
        plt.show()