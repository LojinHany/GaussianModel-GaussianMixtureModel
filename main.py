from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import multivariate_normal as mvn
from sklearn.mixture import GaussianMixture

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

pca = PCA(n_components=20)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

gmm_models = {}
n_components = 3  # number of mixtures per digit (you can experiment: 2–5)

for digit in range(10):
    Xc = x_train_pca[y_train == digit]  # get all samples of this digit
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(Xc)
    gmm_models[digit] = gmm
    
log_likelihoods = np.zeros((x_test_pca.shape[0], 10))

for digit in range(10):
    log_likelihoods[:, digit] = gmm_models[digit].score_samples(x_test_pca)

y_pred = np.argmax(log_likelihoods, axis=1)
GMMaccuracy = np.mean(y_pred == y_test)
print("Empirical Accuracy for Gaussian Mixture Model:", GMMaccuracy)
