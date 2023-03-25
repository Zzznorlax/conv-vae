import os
import glob


def get_latest_ckpt(path):
    try:
        list_of_files = glob.glob(os.path.join(path, '*'))
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file

    except ValueError:
        return None
