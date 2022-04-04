import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import datetime, os, sys

try:
    import clif
except:
    import sys

    sys.path.append("../../")
    # from eof import fingerprints
    import clif
    import clif.preprocessing as cpp

##########################
# Load data
##########################

DATA_DIR = "../../../e3sm_data/fingerprint/"
data = xr.open_dataarray(os.path.join(DATA_DIR, "Temperature.nc"))

##########################
# Pipelining Transforms
##########################
from sklearn.pipeline import Pipeline

pipe = Pipeline(
    steps=[
        (
            "clip",
            cpp.ClipTransform(
                dims=["lat", "plev"], bounds=[(-60.0, 60.0), (5000.0, np.inf)]
            ),
        ),
        ("anom", cpp.SeasonalAnomalyTransform(cycle="month")),
        ("marginalize", cpp.MarginalizeOutTransform(dims=["lat", "lon"])),
        ("detrend", cpp.LinearDetrendTransform()),
        ("flatten", cpp.FlattenData(dims=["plev"])),
        ("transpose", cpp.Transpose(dims=["time", "plev"])),
        ("scale", cpp.ScalerTransform(scale_type="variance")),
    ]
)

data_new = pipe.fit_transform(data)

##########################
# Begin fingerprinting and plotting EOF time-series scores
##########################
# Now we can begin calculating the EOFs
# obtain fingerprints
n_components = 8
fp = clif.fingerprints(n_eofs=n_components, varimax=False)
fp.fit(data_new)

# extract pca fingerprints and convergence diagnostics
eofs_pca = fp.eofs_
explained_variance_ratio = fp.explained_variance_ratio_
eof_time_series = fp.projections_
print(
    "Explained variance ratios for first {0} components:\n".format(n_components),
    explained_variance_ratio,
)

# i conver tcftime series to datetime for plotting with matplotlib
times = data.indexes["time"].to_datetimeindex(unsafe=True)

# add trend lines to eofs
pinatubo_event = datetime.datetime(1991, 6, 15)

# # plot eof's with trend lines before and after event
# # import nc_time_axis # to allow plotting of cftime datetime using matplotlib
# fig, axes = plt.subplots(3, 2, figsize=(10, 8))
# fig.suptitle("EOF scores for {0} using PCA".format("T"), fontsize=20)
# for i, ax in enumerate(axes.flatten()):
#     eof_ts = eof_time_series[:, i]
#     ax.plot(
#         times,
#         eof_ts,
#         label="PC score {0}".format(i + 1),
#         color="C{0}".format(i),
#         alpha=0.6,
#     )
#     ax.axvline(pinatubo_event, color="k", linestyle="--", alpha=0.5)
#     ax.legend(fancybox=True)
#     ax.grid(True)

# plt.show()


################################
# Dynamic Mode Decomposition
################################
# Full time series data
X_full = data_new.values
X_full = X_full - np.mean(X_full, axis=0)  # mean center

# Get X and X' DMD matrices
X = X_full[:-1]
Xp = X_full[1:]

# Let's compute the full A matrix since it's so small
A = np.linalg.pinv(X) @ Xp

# get first DMD mode and projection
w, V = np.linalg.eig(A.T)  # eigenvectors of A^T
dmd1 = np.real(V[:, 0])
tot_var = np.sum(np.var(X_full, axis=0))
dmd1_proj = (dmd1 @ X_full.T) / (np.abs(w[0]) * np.sqrt(tot_var))
eof1 = eofs_pca[0]

# # Version 1
# # test to compute A via SVD
# U, S, Vt = np.linalg.svd(X, full_matrices=False)
# r = 29  # rank
# # A_ml = Vt.T @ np.diag(1.0 / S) @ U.T @ Xp
# Atilde = np.diag(1.0 / S[:r]) @ U[:, :r].T @ Xp @ Vt[:r].T
# l, W = np.linalg.eig(Atilde.T)
# # Phi = Xp.T @ V_r @ np.diag(1.0 / S_r) @ W_r
# Phi_ml = W.T @ np.diag(1.0 / S[:r]) @ U[:, :r].T @ Xp
# # Phi_ml = Vt[:r].T @ W
# Phi_ml = Phi_ml.T  # convert bases to be in column form

# Version Kutz (columnar vs row)
U, S, Vt = np.linalg.svd(X.T, full_matrices=False)
V = Vt.T
r = min(29, U.shape[0])
U_r = U[:, :r]
S_r = S[:r]
V_r = V[:, :r]
# A_kutz = U.T @ Xp.T @ V @ np.diag(1.0 / S)
Atilde = U_r.T @ Xp.T @ V_r @ np.diag(1.0 / S_r)
D, W_r = np.linalg.eig(Atilde)
Phi_kutz = Xp.T @ V_r @ np.diag(1.0 / S_r) @ W_r
# Phi_kutz = U_r @ W_r

raise SystemExit(0)
################################
# Time lagged covariance
################################
m = X.shape[0]
# compute covariance matrix
C0 = (X.T @ X) / (m - 1)
# compute time_lagged covariance matrix
C1 = (X.T @ Xp) / (m - 1)
# Combined
Cnew = C1 @ np.linalg.inv(C0)
wlim, Vlim = np.linalg.eig(Cnew)

# get first DMD mode and projection
lim1 = np.real(Vlim[:, 0])
lim1_proj = (lim1 @ X_full.T) / (np.abs(wlim[0]) * np.sqrt(tot_var))

################################
# plot different types of EOfs
################################

plev = data_new["plev"].values / 100
fig, axes = plt.subplots(2, 1, figsize=[10.5, 7])
plt.subplots_adjust(hspace=0.35)
axes[0].plot(plev, eof1, label="EOF 1")
axes[0].plot(plev, dmd1, label="DMD")
axes[0].plot(plev, lim1, "--", label="LIM")
axes[0].set_xlabel("hPa")
axes[0].set_xscale("log")
axes[0].grid(True)
axes[0].legend()

# fig, ax = plt.subplots(1, figsize=[8.5, 4])
axes[1].plot(times, eof_time_series[:, 0], label="EOF 1 proj")
axes[1].plot(times, dmd1_proj, label="DMD proj")
axes[1].plot(times, lim1_proj, "--", label="LIM proj")
axes[1].set_xlabel("time")
axes[1].grid(True)
axes[1].legend()
