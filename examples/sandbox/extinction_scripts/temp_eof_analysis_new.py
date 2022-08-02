import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import plot_confusion_matrix
import xarray as xr
import datetime, os, sys
from tqdm import tqdm
from time import time
import clif
import glob
import dask

dask.config.set({"array.slicing.split_large_chunks": True})

########################
# configurations
########################
# define the data directory of E3SM historical decks
QOI = "T"
# define source of H* directories
SOURCE_DIR = os.path.join(
    os.getenv("HOME"), "Research", "cldera_data/fingerprinting/extinction/"
)
forcing = "1.0x"
temperature_file = "T_199101_199512.nc"
save_transformations = False
load_transformations = True

########################
# load data
########################
# extract data for a particular forcing
dir_list = glob.glob(SOURCE_DIR + f"/*{forcing}*")
print("loading the following data\n", dir_list)
nruns = len(dir_list)
# use xarray to open the data set and use dask to import as chunks
ext_runs = {}
print("Loading data...")
start = time()
for ii, fi in enumerate(dir_list):
    file_path = os.path.join(fi, temperature_file)
    ds_temp = xr.open_mfdataset(
        file_path, chunks={"time": 1}
    )  # add chunks={"time": 1} for large data files
    ext_runs[ii] = ds_temp
print(f"Total time to load is {time()-start:.3f} seconds")

# combine all runs into a single data set
print("Combining the data")
start = time()
all_temperatures = [run_i[QOI] for i, run_i in ext_runs.items()]
ext_combined = xr.concat(all_temperatures, "run")
T_avg = ext_combined.mean(dim="run")  # average over runs
print(f"Total time to load is {time()-start:.3f} seconds")

print("get area weights...")
lat_lon_weights = ext_runs[0].area

# extract temperature data to begin preprocessing and fingerprinting

###########################
# initialize preprocessing transforms
###########################

clipT = clif.preprocessing.ClipTransform(dims=["lat"], bounds=[(-60.0, 60.0)])
lat_lon_weights_clipped = clipT.fit_transform(lat_lon_weights)

monthlydetrend = clif.preprocessing.SeasonalAnomalyTransform(cycle="month")

intoutT = clif.preprocessing.MarginalizeOutTransform(
    dims=["lat", "lon"], lat_lon_weights=lat_lon_weights_clipped
)

lindetrendT = clif.preprocessing.LinearDetrendTransform()

transformT = clif.preprocessing.Transpose(dims=["time", "lev"])

from sklearn.pipeline import Pipeline

ext_preproc = Pipeline(
    steps=[
        ("clip", clipT),
        ("anom", monthlydetrend),
        ("marginalize", intoutT),
        ("detrend", lindetrendT),
        ("transpose", transformT),
    ]
)

print("Transforming the data...")
start = time()
T_avg_new = ext_preproc.fit_transform(T_avg)
print(f"Total time to transform avg is {time()-start:.3f} seconds")

start = time()
Ti_new = [ext_preproc.fit_transform(Ti) for Ti in all_temperatures]
print(f"Total time to transform ind is {time()-start:.3f} seconds")

if save_transformations:
    print("Saving transformations")
    start = time()
    T_avg_new.to_netcdf(f"T_avg_new_{forcing}.nc")
    for ii, Ti in enumerate(Ti_new):
        Ti.to_netcdf(f"T{ii}_new_{forcing}.nc")
    print(f"Total time to save transforms is {time()-start:.3f} seconds")

#####################################################################
## Begin fingerprinting and plotting EOF time-series scores
######################################################################
if load_transformations:
    # load data
    T_avg_new = xr.open_dataarray(f"T_avg_new_{forcing}.nc")
    Ti_new = [xr.open_dataarray(f"T{ii}_new_{forcing}.nc") for ii in range(nruns)]

# Now we can begin calculating the EOFs
# obtain fingerprints
n_components = 10
fp = clif.fingerprints(
    n_eofs=n_components,
    method="pca",
    method_opts={"whiten": True, "svd_solver": "arpack"},
    varimax=False,
)

# Fit EOF to deck avg data
print("Computing fingerprints...")
start = time()
fp.fit(T_avg_new)
evr_ = fp.explained_variance_ratio_
print("Total time is {0:.3f} seconds".format(time() - start))

# # plot eofs:
# fig, ax = plt.subplots(1, figsize=[12, 6])
# ax.plot(T_avg_new["lev"].values, fp.eofs_[0], linewidth=3)  # lev in pascals
# # ax.axvspan(10, 100, alpha=0.15, color="C5", label="lower stratosphere")
# ax.grid(True, which="both", linestyle="--", alpha=0.6)
# ax.set_xscale("log")
# ax.set_xlabel("hPa", fontsize=14)
# ax.set_ylabel("Scaled \n principal \ncomponent", rotation=0, labelpad=35, fontsize=14)
# ax.legend(fancybox=True, fontsize=14)
# ax.invert_xaxis()
# fig.savefig(f"pc1_{forcing}.png")

# transforming via principal components
print("Transforming data...")
times = T_avg_new["time"].indexes["time"].to_datetimeindex(unsafe=True)
T_avg_proj = fp.transform(T_avg_new.values)

# # projects individual deck ensemble runs onto EOFs
# Ti_projs = []
# for Ti_temp in Ti_new:
#     Ti_projs.append(fp.transform(Ti_temp.values))

# plot projections
plot_comp = 0
fig2, ax2 = plt.subplots(1, figsize=(11, 5))
ax2.plot(
    times,
    T_avg_proj[:, plot_comp],
    label="E3SMv1",
    color="C0",
    linewidth=2,
    alpha=0.8,
)
# ax2.plot(times, np.array([ti[:, plot_comp] for ti in Ti_projs]).T, alpha=0.3)
# fig2.savefig(f"proj1_{forcing}.png")


# load other data
T_avg_new_1p5 = xr.open_dataarray(f"T_avg_new_1.5x.nc")
T_avg_new_0p5 = xr.open_dataarray(f"T_avg_new_0.5x.nc")
T_avg_new_0p0 = xr.open_dataarray(f"T_avg_new_0.0x.nc")

T_avg_proj_1p0 = fp.transform(T_avg_new.values)
T_avg_proj_1p5 = fp.transform(T_avg_new_1p5.values)
T_avg_proj_0p5 = fp.transform(T_avg_new_0p5.values)
T_avg_proj_0p0 = fp.transform(T_avg_new_0p0.values)

# plot projections
plot_comp = 0
fig3, ax3 = plt.subplots(1, figsize=(11, 5))
ax3.plot(times, T_avg_proj_1p0[:, plot_comp], label="1.0x forcing")
ax3.plot(times, T_avg_proj_1p5[:, plot_comp], label="1.5x forcing")
ax3.plot(times, T_avg_proj_0p5[:, plot_comp], label="0.5x forcing")
ax3.plot(times, T_avg_proj_0p0[:, plot_comp], label="0.0x forcing")
ax3.legend(fancybox=True)
ax3.grid(True,which='both')
ax3.set_ylabel("EOF \nscores",rotation=0, labelpad=15)
fig3.savefig("proj_comparison.png")
