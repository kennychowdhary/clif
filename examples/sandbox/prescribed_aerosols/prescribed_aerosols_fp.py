import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import xarray as xr
import datetime, os, sys
from tqdm import tqdm
from time import time
import clif
import glob
import dask

###############################
# Define data directories
###############################
# define source directory of data
SOURCE_DIR = os.path.join(
    os.getenv("HOME"), "Research", "cldera_data/fingerprinting/prescribed_aerosols"
)

# e3sm runs
RUN_DIR = ["run_01", "run_02", "run_03", "run_04", "run_06"]

save_data = False

###############################
# Load QOI data and avg runs
###############################
# get file list for TREFHT
QOI = "TREFHT"

file_paths = []
for rundir in RUN_DIR:
    file_list_temp = glob.glob(os.path.join(SOURCE_DIR, rundir) + f"/{QOI}*.nc")
    file_paths += file_list_temp

# load datasets and extract data arrays
da_avg = xr.open_dataset(file_paths[0])[QOI]
for file_path in file_paths[1:]:
    da_avg += xr.open_dataset(file_path)[QOI]
da_avg /= len(file_paths)

# get lat lon area weight as well
area_weights = xr.open_dataset(file_paths[0])["area"]
area_weights /= area_weights.sum()

lat, lon = da_avg["lat"], da_avg["lon"]

###############################
# Preprocess data using clif transforms
###############################
clipT = clif.preprocessing.ClipTransform(dims=["lat"], bounds=[(-60.0, 60.0)])
area_weights_clp = clipT.fit_transform(area_weights)
lat_clp = clipT.fit_transform(lat)

monthlydetrend = clif.preprocessing.SeasonalAnomalyTransform(cycle="month")

flattenT = clif.preprocessing.FlattenData(dims=["lat", "lon"])

lindetrendT = clif.preprocessing.LinearDetrendTransform()

transformT = clif.preprocessing.Transpose(dims=["time", "lat_lon"])

from sklearn.pipeline import Pipeline

da_preproc = Pipeline(
    steps=[
        ("clip", clipT),
        ("anom", monthlydetrend),
        ("detrend", lindetrendT),
        ("flatten", flattenT),
        ("transpose", transformT),
    ]
)

da_avg_new = da_preproc.fit_transform(da_avg)

if save_data:
    np.save(f"{QOI}_avg_new", da_avg_new.values)
    raise SystemExit(0)

###############################
# Compute fingerpring of processes data
###############################

# Now we can begin calculating the EOFs
# obtain fingerprints
n_components = 10
fp = clif.fingerprints(
    n_eofs=n_components,
    method="pca",
    method_opts={"whiten": True, "svd_solver": "arpack"},
    varimax=False,
)

# Fit EOF to avg data
print("Computing fingerprints...")
start = time()
fp.fit(da_avg_new)
evr_ = fp.explained_variance_ratio_
print("Exp. var ratio:\n", evr_)
print("Total time is {0:.3f} seconds".format(time() - start))

# get EOFs
eofs_ = fp.eofs_
pc1 = eofs_[0].reshape(len(lat_clp), len(lon))
pc1_xr = xr.DataArray(pc1, coords={"lat": lat_clp, "lon": lon})

# get EOF scores
times = da_avg_new["time"].indexes["time"].to_datetimeindex(unsafe=True)
score1 = fp.transform(da_avg_new.values)
plt.plot(np.load("times_for_plotting.npy"), score1[:, 0])
plt.title(QOI)

raise SystemExit(0)

###############################
# plot the eof
###############################
# import the lat lon contour plotting class
import clif.visualization as cviz

# Now we initialize the contout.plot_lat_lon class with some parameters like the color map and titles
eofplot = cviz.contour.plot_lat_lon(
    cmap_name="e3sm_default",
    title="Temperature",
    rhs_title="\u00b0" + "K",
    lhs_title="E3SMv2 ne30np4",
)

# eofplot.show(da_avg.mean(dim="time"))
eofplot.show(pc1_xr)
