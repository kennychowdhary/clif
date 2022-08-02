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
from tqdm import tqdm

###############################
# Define data directories
###############################
# define source directory of data
SOURCE_DIR = os.path.join(
    os.getenv("HOME"), "Research", "cldera_data/fingerprinting/historical_runs"
)

# e3sm runs
RUN_DIR = ["run_151", "run_201", "run_251", "run_301"]

# QOI
QOI = "TREFHT"

# get list of file names for a single run
run_num = RUN_DIR[0]
file_paths = []
file_names_full = glob.glob(os.path.join(SOURCE_DIR, run_num) + f"/{QOI}*.nc")
file_names = [os.path.basename(fn) for fn in file_names_full]

###############################
# compute averages
###############################

# make average run directory
AVG_DIR = "run_avg"
os.makedirs(os.path.join(SOURCE_DIR, AVG_DIR), exist_ok=True)


for fn in tqdm(file_names):
    # load datasets and extract data arrays
    fn_path_temp = os.path.join(SOURCE_DIR, RUN_DIR[0], fn)
    da_avg_temp = xr.open_dataset(fn_path_temp)[QOI]
    for run_i in RUN_DIR[1:]:
        fn_path_temp = os.path.join(SOURCE_DIR, run_i, fn)
        da_avg_temp += xr.open_dataset(fn_path_temp)[QOI]
    da_avg = da_avg_temp / len(RUN_DIR)
    # save to new average directory
    da_avg.to_netcdf(os.path.join(SOURCE_DIR, AVG_DIR, fn))
