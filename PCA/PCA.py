import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean Centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Covarience; fn needs samples as col
        cov = np.cov(X.T)

        # Eigen Ventor or Eigen Values
        eigen_values, eigen_vectors = np.linalg.eig(cov)

        # Transpose eigen vector
        eigen_vectors = eigen_vectors.T

        # Sort eigen vectors
        idxs = np.argsort(eigen_values)[ : : -1]
        eigen_values = eigen_values[idxs]
        eigen_vectors = eigen_vectors[idxs]    

        self.components = eigen_vectors[ : self.n_components]

    def transform(self, X):
        # Project Data
        X = X - self.mean
        return np.dot(X, self.components.T)
    
if __name__ == "__main__":
    # imports libs
    from sklearn import datasets as df
    import matplotlib.pyplot as plt
    # import matplotlib.cm as cm

    #data = df.load_digits()
    data = df.load_iris()
    X = data.data
    y = data.target

    # Project the data onto the primary PC
    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("shape of X: ", X.shape)
    print("shape of X_projected: ", X_projected.shape)

    x1 = X_projected[ :, 0]
    x2 = X_projected[ :, 1]
    # Scatter plot
    plt.scatter(x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.get_cmap("viridis", 3))

    # plt.scatter(x1, x2, c = y, edgecolor = "none", alpha = 0.8, cmap = plt.cm.get_cmap("viridis", 3))

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show