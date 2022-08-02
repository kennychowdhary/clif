import datetime, os, sys

run_numbers = [201, 301, 151, 251]
save_dir = "/ascldap/users/kchowdh/Research/cldera_data/historical_runs"


def get_data():

    for run_number in run_numbers:
        file_dir = f"/projects/cldera/data/E3SM/E3SMv2-simulation-campaign/v2.LR.historical/v2.LR.historical_0{run_number}/post/atm/180x360_aave/ts/monthly/5yr"

        # get files and full paths
        file_path1 = os.path.join(file_dir, "FSNT_*.nc")
        file_path2 = os.path.join(file_dir, "TREFHT_*.nc")
        new_folder = os.path.join(
                save_dir, f"run_{str(run_number).zfill(2)}"
            )
        os.makedirs(new_folder, exist_ok=True)
        for file_path in [file_path1,file_path2]:
            print(f"Copying {file_path} to {new_folder}")
            os.system(f"cp {file_path} {new_folder}/")
