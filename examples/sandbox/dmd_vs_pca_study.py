from ast import Lambda
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    Parameters
    ----------
    u : ndarray
        u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
    v : ndarray
        u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
        The input v should really be called vt to be consistent with scipy's
        output.
    u_based_decision : bool, default=True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.
    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v


def dmd_kutz(X_full, rank=8):
    """SVD-based method for computing DMD

    See Algorithm 1.1 in Kutz's DMD book

    Only difference is that our input matrix has snapshots in the row index.
    Thus, all data matrices are transposed in Kutz's algorithm. This is to better
    align with ML and PCA.
    """
    # Get X and X' DMD matrices
    X1 = X_full[:-1]
    X2 = X_full[1:]

    # Version Kutz (columnar vs row)
    U, S, Vt = np.linalg.svd(X1.T, full_matrices=False)
    U, Vt = svd_flip(U, Vt)  # consistent sign determination
    V = Vt.T

    # full rank
    r = rank  # U.shape[0]

    # shortened matrices
    U_r = U[:, :r]
    S_r = S[:r]
    V_r = V[:, :r]

    Atilde = U_r.conj().T @ X2.T @ V_r @ np.diag(1.0 / S_r)
    D, W_r = np.linalg.eig(Atilde)
    Phi = X2.T @ V_r @ np.diag(1.0 / S_r) @ W_r
    V_dmd = Phi.T  # for direct comparison to PCA

    # compute recovered signal
    x1 = X1[0]
    b = np.linalg.pinv(Phi) @ x1  # OLS solution
    mm1 = X1.shape[0]
    time_dynamics = []
    for iter in range(mm1 + 1):
        time_dynamics.append(b * np.exp(np.log(D) * iter))
    Xdmd = Phi @ np.array(time_dynamics).T
    Xr_dmd = Xdmd.T  # for direct comparison to PCA

    # return matrix where rows are the modes
    return V_dmd, Xr_dmd


# Generate data of a traveling wave
x = np.linspace(0, 5, 250)
sig = 0.25
wave = lambda t: np.exp(-((x - t) ** 2) / 2.0 / sig ** 2) / np.sqrt(
    2 * np.pi * sig ** 2
)

X = []
t = np.linspace(1, 4, 128)
dt = np.diff(t)[0]
for ti in t:
    X.append(wave(ti))
    # plt.plot(t, wave(mu))
X = np.array(X)

# compute PCA on the traveling wave data matrix
pca = PCA(n_components=8)
Xhat = pca.fit_transform(X)
V_pca = pca.components_
Xr_pca = pca.inverse_transform(Xhat)

# dynamic mode decomposition
V_dmd, Xr_dmd = dmd_kutz(X, rank=8)
