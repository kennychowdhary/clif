import datetime, os, sys

run_numbers = [1, 2, 3, 4, 6]
save_dir = "/ascldap/users/kchowdh/Research/cldera_data/extinction"


def get_data(forcing="1.0x"):
    if forcing == "1.0x":
        f_str = "_"
    elif forcing == "1.5x":
        f_str = "_e1.5_"
    elif forcing == "0.5x":
        f_str = "_e0.5_"
    elif forcing == "0.0x":
        f_str = "_e0.0_"

    for run_number in run_numbers:
        file_dir = f"/projects/cldera/data/E3SM/e3sm-cldera/le/v2.LR.WCYCL20TR_01{run_number}1_b1991{f_str}cori/post/atm/180x360_aave/ts/monthly/5yr"
        file_path = os.path.join(file_dir, "T_199101_199512.nc")
        new_folder = os.path.join(
            save_dir, f"forcing_{forcing}_{str(run_number).zfill(2)}"
        )
        print(f"Copying {file_path} to {new_folder}")
        os.makedirs(new_folder, exist_ok=True)
        os.system(f"cp {file_path} {new_folder}/")
