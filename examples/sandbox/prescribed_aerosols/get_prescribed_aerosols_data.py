import datetime, os, sys

run_numbers = [1, 2, 3, 4, 6]
save_dir = "/ascldap/users/kchowdh/Research/cldera_data/prescribed_aerosols"


def get_data():

    for run_number in run_numbers:
        file_dir = f"/projects/cldera/data/E3SM/e3sm-cldera/le/v2.LR.WCYCL20TR_01{run_number}1_b1991_cori/post/atm/180x360_aave/ts/monthly/5yr"

        # get files and full paths
        file_path1 = os.path.join(file_dir, "FSNT_199101_199512.nc")
        file_path2 = os.path.join(file_dir, "TREFHT_199101_199512.nc")
        new_folder = os.path.join(
                save_dir, f"run_{str(run_number).zfill(2)}"
            )
        os.makedirs(new_folder, exist_ok=True)
        for file_path in [file_path1,file_path2]:
            print(f"Copying {file_path} to {new_folder}")
            os.system(f"cp {file_path} {new_folder}/")
