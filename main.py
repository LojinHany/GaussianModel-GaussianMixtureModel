from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import multivariate_normal as mvn
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

class GMM:
    def __init__(self, n_components, n_iters=50):
        self.k = n_components
        self.n_iters = n_iters

    def fit(self, X):
        n_samples, n_features = X.shape
        np.random.seed(42)
        random_idx = np.random.choice(n_samples, self.k, replace=False)
        self.means = X[random_idx]
        self.covs = np.array([np.cov(X.T) + np.eye(n_features)*1e-3 for _ in range(self.k)])
        self.weights = np.ones(self.k) / self.k

        for _ in range(self.n_iters):
            resp = np.zeros((n_samples, self.k))
            for i in range(self.k):
                pdf_vals = mvn.pdf(X, self.means[i], self.covs[i], allow_singular=True)
                resp[:, i] = self.weights[i] * pdf_vals

            resp_sum = resp.sum(axis=1, keepdims=True)
            resp_sum[resp_sum == 0] = 1e-12
            resp = resp / resp_sum

            N_k = resp.sum(axis=0)
            for i in range(self.k):
                if N_k[i] < 1e-8:
                    continue
                self.means[i] = (resp[:, i].reshape(-1, 1) * X).sum(axis=0) / N_k[i]
                diff = X - self.means[i]
                self.covs[i] = (resp[:, i].reshape(-1, 1) * diff).T @ diff / N_k[i]
                self.covs[i] += np.eye(n_features) * 1e-3
                self.weights[i] = N_k[i] / n_samples

        self.covs = np.nan_to_num(self.covs, nan=1e-3, posinf=1e-3, neginf=1e-3)

    def score_samples(self, X):
        probs = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            probs[:, i] = self.weights[i] * mvn.pdf(X, self.means[i], self.covs[i], allow_singular=True)
        probs[probs == 0] = 1e-12
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
        log_likelihoods[:, i] = mvn.logpdf(x_test, means[i], covs[i], allow_singular=True)
    indices = np.argmax(log_likelihoods, axis=1)
    y_pred = np.array([class_labels[i] for i in indices])
    return y_pred, log_likelihoods


mnist = fetch_openml('mnist_784', version=1, as_frame=False)
x, y = mnist.data, mnist.target.astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

pca, class_labels, means, covs = trainGaussianModel(x_train, y_train)
y_pred, log_likelihoods = testGaussianModel(x_test, pca, class_labels, means, covs)
accuracy = np.mean(y_pred == y_test)
print("Empirical Accuracy For Gaussian Model:", accuracy)

x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

n_components = 3
gmms = {}
for digit in range(10):
    Xc = x_train_pca[y_train == digit]
    gmm = GMM(n_components=n_components, n_iters=30)
    gmm.fit(Xc)
    gmms[digit] = gmm

log_likelihoods_gmm = np.zeros((x_test_pca.shape[0], 10))
for digit in range(10):
    log_likelihoods_gmm[:, digit] = gmms[digit].score_samples(x_test_pca)

y_pred = np.argmax(log_likelihoods_gmm, axis=1)
accuracy = np.mean(y_pred == y_test)
print("Empirical Accuracy for your Gaussian Mixture Model:", accuracy)

y_test_bin = label_binarize(y_test, classes=range(10))

plt.figure(figsize=(10, 8))
for i in range(10):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], log_likelihoods[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.title('ROC Curves for Gaussian Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

plt.figure(figsize=(10, 8))
for i in range(10):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], log_likelihoods_gmm[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.title('ROC Curves for Gaussian Mixture Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()
