"""Classical machine learning models for anomaly detection benchmarking.

Models implemented:
    - IsolationForestModel:     Tree-based anomaly isolation
    - OneClassSVMModel:         Support vector novelty detection
    - LOFModel:                 Local Outlier Factor (density-based)
    - PCAReconstructionModel:   PCA-based reconstruction error

All models follow a common interface:
    - fit(X_train)          → train on normal data (numpy arrays)
    - anomaly_score(X)      → per-sample scores (higher = more anomalous)
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================================
# BASE CLASS
# ============================================================================

class BaseClassicalModel:
    """Base interface for classical anomaly detection models."""

    name: str = "BaseClassical"

    def fit(self, X_train: np.ndarray) -> "BaseClassicalModel":
        """Fit the model on normal training data."""
        raise NotImplementedError

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample anomaly scores.  Higher = more anomalous."""
        raise NotImplementedError


# ============================================================================
# 1. ISOLATION FOREST
# ============================================================================

class IsolationForestModel(BaseClassicalModel):
    """Isolation Forest for anomaly detection.

    Isolates anomalies by randomly selecting features and split values.
    Anomalies are easier to isolate → shorter path in the tree → higher score.

    Strengths: fast training, handles high-dimensional data well, no
    distance metric needed.
    """

    name = "Isolation Forest"

    def __init__(self, n_estimators=200, contamination='auto', random_state=42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_train):
        self.model.fit(X_train)
        return self

    def anomaly_score(self, X):
        # score_samples: lower = more anomalous → negate for consistency
        return -self.model.score_samples(X)


# ============================================================================
# 2. ONE-CLASS SVM
# ============================================================================

class OneClassSVMModel(BaseClassicalModel):
    """One-Class SVM for novelty detection.

    Learns a decision boundary around normal data in kernel space.
    Points outside the boundary are classified as anomalies.

    Note: Subsamples training data to max_samples for computational
    efficiency — SVM training is O(n²) in the number of samples.
    """

    name = "One-Class SVM"

    def __init__(self, kernel='rbf', gamma='scale', nu=0.1, max_samples=10_000):
        self.model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        self.scaler = StandardScaler()
        self.max_samples = max_samples

    def fit(self, X_train):
        # Subsample for speed — SVM is O(n²) in samples
        if len(X_train) > self.max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_train), self.max_samples, replace=False)
            X_train = X_train[idx]
            print(f"    (OC-SVM: subsampled to {self.max_samples:,} training samples)")

        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled)
        return self

    def anomaly_score(self, X):
        X_scaled = self.scaler.transform(X)
        # decision_function: negative = anomalous → negate
        return -self.model.decision_function(X_scaled)


# ============================================================================
# 3. LOCAL OUTLIER FACTOR (LOF)
# ============================================================================

class LOFModel(BaseClassicalModel):
    """Local Outlier Factor for novelty detection.

    Measures local density deviation of a data point relative to its
    neighbours.  Points with substantially lower density than their
    neighbours are considered anomalies.

    Note: Uses novelty=True mode so it can score new (unseen) data.
    Subsamples training data for speed on large datasets.
    """

    name = "LOF"

    def __init__(self, n_neighbors=20, contamination='auto', max_samples=15_000):
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,   # required for decision_function on new data
            n_jobs=-1,
        )
        self.max_samples = max_samples

    def fit(self, X_train):
        if len(X_train) > self.max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_train), self.max_samples, replace=False)
            X_train = X_train[idx]
            print(f"    (LOF: subsampled to {self.max_samples:,} training samples)")

        self.model.fit(X_train)
        return self

    def anomaly_score(self, X):
        # decision_function: negative = anomalous → negate
        return -self.model.decision_function(X)


# ============================================================================
# 4. PCA RECONSTRUCTION ERROR
# ============================================================================

class PCAReconstructionModel(BaseClassicalModel):
    """PCA-based anomaly detection using reconstruction error.

    Projects data into a lower-dimensional principal component subspace
    and reconstructs it back.  Normal data reconstructs well; anomalies
    produce high reconstruction error because they lie outside the learned
    subspace.

    Anomaly score = per-sample mean absolute reconstruction error.
    """

    name = "PCA Reconstruction"

    def __init__(self, n_components=0.95):
        """
        Args:
            n_components: Number of components or variance ratio to retain.
                          0.95 = keep enough components to explain 95% of variance.
        """
        self.n_components = n_components
        self.pca = None
        self.scaler = StandardScaler()

    def fit(self, X_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_scaled)
        print(f"    (PCA: retained {self.pca.n_components_} components, "
              f"explained variance = {self.pca.explained_variance_ratio_.sum():.3f})")
        return self

    def anomaly_score(self, X):
        X_scaled = self.scaler.transform(X)
        X_projected = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_projected)
        # Per-sample MAE as anomaly score
        return np.abs(X_scaled - X_reconstructed).mean(axis=1)
