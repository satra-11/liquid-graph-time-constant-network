import vmas
import pathlib


if __name__ == "__main__":
    init_py_path = pathlib.Path(vmas.__file__)
    print(init_py_path.parent)