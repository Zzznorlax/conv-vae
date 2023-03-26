import os
import sys
import shutil
import glob

from pathlib import Path
from os import listdir
from os.path import isfile
from os.path import join
from typing import List


def get_extension(file_path: str):
    return os.path.splitext(file_path)


def get_filename(file_path):
    return os.path.basename(file_path)


def get_current_dir():
    return sys.path[0]


def get_local_file(filename):
    return os.path.join(sys.path[0], filename)


def get_files(folder_path, ext: List[str] = []):
    files = []

    for _ext in ext:
        files += glob.glob(join(folder_path, "**", f"*{_ext}"), recursive=True)

    return files


def get_sorted_files(folder_path, ext: List[str] = []):

    path_list = get_files(folder_path, ext)

    def filename_key(x: str):
        filename, _ = get_extension(get_filename(x))
        filename = filename.zfill(4)
        return filename

    path_list.sort(key=filename_key)

    return path_list


def get_folders(dir: str):
    return [join(dir, f) for f in listdir(dir) if not isfile(join(dir, f))]


def batch_rename(filenames: List[str], old_name: str, new_name: str):
    for filename in filenames:
        new_filename = filename.replace(old_name, new_name)
        os.rename(filename, new_filename)


def list_files(folder_path):
    files = get_files(folder_path)
    for file in files:
        print(file)
    return files


def clear_folder(folder_path):
    files = get_files(folder_path)
    for file in files:
        os.unlink(os.path.join(folder_path, file))


def move(origin_path, target_path):
    os.replace(origin_path, target_path)


def get_parent_folder(folder_path):
    return Path(folder_path).parent


def clear(dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)

        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)

            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def create_folder(dir, folder_name):
    folder_path = join(dir, folder_name)
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    return folder_path


if __name__ == '__main__':

    parent_folder_path = ""

    folders = get_folders(parent_folder_path)
    batch_rename(folders, "test", "new_test")
