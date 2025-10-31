from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import multivariate_normal as mvn


class GMM:
    def __init__(self, n_components, n_iters=50):
        self.k = n_components
        self.n_iters = n_iters

    def fit(self, X):
        n_samples, n_features = X.shape

        # Initialize parameters
        np.random.seed(42)
        random_idx = np.random.choice(n_samples, self.k, replace=False)
        self.means = X[random_idx]
        self.covs = np.array([np.cov(X.T) for _ in range(self.k)])
        self.weights = np.ones(self.k) / self.k

        # EM algorithm
        for _ in range(self.n_iters):
            # E-step
            resp = np.zeros((n_samples, self.k))
            for i in range(self.k):
                resp[:, i] = self.weights[i] * mvn.pdf(X, self.means[i], self.covs[i])
            resp = resp / resp.sum(axis=1, keepdims=True)

            # M-step
            N_k = resp.sum(axis=0)
            for i in range(self.k):
                self.means[i] = (resp[:, i].reshape(-1, 1) * X).sum(axis=0) / N_k[i]
                diff = X - self.means[i]
                self.covs[i] = (resp[:, i].reshape(-1, 1) * diff).T @ diff / N_k[i]
                self.covs[i] += np.eye(n_features) * 1e-6
                self.weights[i] = N_k[i] / n_samples

    def score_samples(self, X):
        probs = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            probs[:, i] = self.weights[i] * mvn.pdf(X, self.means[i], self.covs[i])
        return np.log(probs.sum(axis=1))
    
def trainGaussianModel(x_train, y_train):
    class_labels = np.unique(y_train)
    pca = PCA(n_components=20)
    pca.fit(x_train)
    x_train = pca.transform(x_train)

    means = []
    covs = []
    for c in class_labels:
        Xc = x_train[y_train == c]
        mu = np.mean(Xc, axis=0)
        cov = np.cov(Xc.T)
        cov += np.eye(cov.shape[0]) * 1e-6
        means.append(mu)
        covs.append(cov)
    return pca, class_labels, means, covs


def testGaussianModel(x_test, pca, class_labels, means, covs):
    x_test = pca.transform(x_test)
    n_samples = x_test.shape[0]
    n_classes = len(class_labels)
    log_likelihoods = np.zeros((n_samples, n_classes))

    for i in range(n_classes):
        log_likelihoods[:, i] = mvn.logcdf(x_test, means[i], covs[i])

    indices = np.argmax(log_likelihoods, axis=1)
    y_pred = []
    for i in indices:
        y_pred.append(class_labels[i])
    y_pred = np.array(y_pred)

    return y_pred


mnist = fetch_openml('mnist_784', version=1, as_frame=False)
x, y = mnist.data, mnist.target.astype(int)

# Split data (70% train, 30% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train
pca, class_labels, means, covs = trainGaussianModel(x_train, y_train)

# Test
y_pred = testGaussianModel(x_test, pca, class_labels, means, covs)

accuracy = np.mean(y_pred == y_test)
print("Empirical Accuracy For Gaussian Model:", accuracy)

n_components = 3  # you can change to 2, 4, etc.
gmms = {}
for digit in range(10):
    Xc = x_train[y_train == digit]
    gmm = GMM(n_components=n_components, n_iters=30)
    gmm.fit(Xc)
    gmms[digit] = gmm

log_likelihoods = np.zeros((x_test.shape[0], 10))
for digit in range(10):
    log_likelihoods[:, digit] = gmms[digit].score_samples(x_test)

y_pred = np.argmax(log_likelihoods, axis=1)
accuracy = np.mean(y_pred == y_test)
print("Empirical Accuracy for your Gaussian Mixture Model:", accuracy)
