import numpy as np

class PLS:
    def __init__(self, ncomp, weights=None):
        self.ncomp = ncomp
        self.weights = weights

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        n, zp = X.shape
        q = Y.shape[1]

        if self.weights is None:
            self.weights = np.ones(n, dtype=np.float64) / n
        else:
            self.weights = np.asarray(self.weights, dtype=np.float64)
            self.weights = self.weights / np.sum(self.weights)

        self.xmeans = np.sum(self.weights[:, None] * X, axis=0)
        X = X - self.xmeans

        self.ymeans = np.sum(self.weights[:, None] * Y, axis=0)
        Y = Y - self.ymeans

        self.T = np.zeros((n, self.ncomp), dtype=np.float64)
        self.R = np.zeros((zp, self.ncomp), dtype=np.float64)
        self.W = np.zeros((zp, self.ncomp), dtype=np.float64)
        self.P = np.zeros((zp, self.ncomp), dtype=np.float64)
        self.C = np.zeros((q, self.ncomp), dtype=np.float64)
        self.TT = np.zeros(self.ncomp, dtype=np.float64)

        Xd = self.weights[:, None] * X
        tXY = Xd.T @ Y

        for a in range(self.ncomp):
            if q == 1:
                w = tXY[..., 0]
            else:
                # SVD for multi-target Y
                u, _, _ = np.linalg.svd(tXY.T, full_matrices=False)
                u = u[:, 0]
                w = tXY @ u

            w = w / np.sqrt(np.sum(w * w))

            r = w.copy()
            if a > 0:
                for j in range(a):
                    r = r - np.sum(self.P[:, j] * w) * self.R[:, j]

            t = X @ r
            tt = np.sum(self.weights * t * t)

            c = (tXY.T @ r) / tt
            p = (Xd.T @ t) / tt

            tXY = tXY - (p[:, None] @ c[None]) * tt

            self.T[:, a] = t
            self.P[:, a] = p
            self.W[:, a] = w
            self.R[:, a] = r
            self.C[:, a] = c
            self.TT[a] = tt

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = X - self.xmeans
        T_new = X @ self.R
        return T_new

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.T

    def get_params(self):
        return {
            "T": self.T,
            "P": self.P,
            "W": self.W,
            "C": self.C,
            "R": self.R,
            "TT": self.TT,
            "xmeans": self.xmeans,
            "ymeans": self.ymeans,
            "weights": self.weights,
            "T.ortorcho": True
        }

    def predict(self, X, nlv=None):
        X = np.asarray(X, dtype=np.float64)
        X = X - self.xmeans

        if nlv is None:
            nlv = self.ncomp
        else:
            nlv = min(nlv, self.ncomp)

        B = self.W[:, :nlv] @ np.linalg.inv(self.P[:, :nlv].T @ self.W[:, :nlv]) @ self.C[:, :nlv].T
        predictions = X @ B + self.ymeans
        return predictions
    
    
def rosa_pls(X_blocks, Y, ncomp_list):
    """
    ROSA-PLS for multiple X blocks using your NumPy PLS class.
    X_blocks: list of np.ndarray, each (n_samples, n_features_block)
    Y: np.ndarray, shape (n_samples, n_targets)
    ncomp_list: list of int, number of components for each block
    Returns: list of dicts with PLS models and predictions for each block
    """
    Y_res = Y.copy()
    X_res_blocks = [X.copy() for X in X_blocks]
    results = []

    for i, (X, ncomp) in enumerate(zip(X_blocks, ncomp_list)):
        pls = PLS(ncomp=ncomp)
        pls.fit(X_res_blocks[i], Y_res)
        Y_pred = pls.predict(X_res_blocks[i], ncomp)
        results.append({
            'pls': pls,
            'Y_pred': Y_pred,
            'block': i
        })
        # Update Y residual for next block
        Y_res = Y_res - Y_pred
        # Orthogonalize next X blocks with respect to current block's scores
        if i < len(X_blocks) - 1:
            T = pls.T[:, :ncomp]  # scores
            for j in range(i+1, len(X_blocks)):
                # Project and remove the part explained by T
                P = np.linalg.pinv(T) @ X_res_blocks[j]
                X_res_blocks[j] = X_res_blocks[j] - T @ P

    return results