import zipfile
import pathlib
import subprocess

subprocess.run("config.py")
DATASETS_DIR = globals().get("DATASETS_DIR")

zip_path = pathlib.Path(DATASETS_DIR.joinpath("64x64_SIGNS.zip"))
zip_file = zipfile.ZipFile(zip_path)
zip_file.extractall(DATASETS_DIR.joinpath("64x64_SIGNS"))
zip_file.close()
