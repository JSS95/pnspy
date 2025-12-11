from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildF2PyExt(build_ext):
    """Custom build_ext to compile Fortran code with f2py (Meson backend)."""

    def run(self):
        import glob
        import shutil
        import subprocess
        import sys

        # Compile Fortran module using f2py
        f2py_cmd = [
            sys.executable,
            "-m",
            "numpy.f2py",
            "-c",
            "src/minpack/lmder.f",
            "-m",
            "_lmder",
            "--backend",
            "meson",
        ]
        subprocess.check_call(f2py_cmd)

        for file_path in glob.glob("_lmder.*"):
            shutil.move(file_path, "src/pns")


setup(
    ext_modules=[Extension("pns._lmder", sources=["src/minpack/lmder.f"])],
    cmdclass={"build_ext": BuildF2PyExt},
)
