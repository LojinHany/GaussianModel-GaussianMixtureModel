from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import multivariate_normal as mvn


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
