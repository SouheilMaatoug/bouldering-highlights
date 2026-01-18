import os
import tempfile


def create_tmp_file(suffix: str):
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    return tmp.name


def clear_file(filename: str):
    os.remove(filename)
