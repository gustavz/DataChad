import io
import os
import shutil
from pathlib import Path

from datachad.backend.constants import DATA_PATH
from datachad.backend.logging import logger
from datachad.backend.utils import clean_string_for_storing


def concatenate_file_names(strings: list[str], n_max: int = 30) -> str:
    # Calculate N based on the length of the list
    n = max(1, n_max // len(strings))
    result = ""
    # Add up the first N characters of each string
    for string in sorted(strings):
        result += f"-{string[:n]}"
    return clean_string_for_storing(result)


def get_data_source_and_save_path(
    files: list[io.BytesIO],
) -> tuple[str, Path]:
    # generate data source string and path to save files to
    if len(files) > 1:
        # we create a folder name by adding up parts of the file names
        path = DATA_PATH / concatenate_file_names([f.name for f in files])
        data_source = path
    else:
        path = DATA_PATH
        data_source = path / files[0].name
    if not os.path.exists(path):
        os.makedirs(path)
    return str(data_source), path


def save_file(file: io.BytesIO, path: Path) -> None:
    # save streamlit UploadedFile to path
    file_path = str(path / file.name)
    file.seek(0)
    file_bytes = file.read()
    file = open(file_path, "wb")
    file.write(file_bytes)
    file.close()
    logger.info(f"Saved: {file_path}")


def save_files(files: list[io.BytesIO]) -> str:
    # streamlit uploaded files need to be stored locally
    # before embedded and uploaded to the hub
    if not files:
        return None
    data_source, save_path = get_data_source_and_save_path(files)
    for file in files:
        save_file(file, save_path)
    return data_source


def delete_files(files: list[io.BytesIO]) -> None:
    # cleanup locally stored files
    # the correct path is stored to data_source
    if not files:
        return
    data_source, _ = get_data_source_and_save_path(files)
    if os.path.isdir(data_source):
        shutil.rmtree(data_source)
    elif os.path.isfile(data_source):
        os.remove(data_source)
    else:
        return
    logger.info(f"Removed: {data_source}")
