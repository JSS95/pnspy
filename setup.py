import subprocess
from pathlib import Path

import numpy
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class F2PyBuildExt(build_ext):
    def build_extension(self, ext):
        fortran_src = Path("src/minpack/lmder.f")
        build_dir = Path(self.build_temp)
        build_dir.mkdir(parents=True, exist_ok=True)

        # f2py output file
        generated_c = build_dir / "lmdermodule.c"

        # Run f2py manually
        cmd = [
            "python",
            "-m",
            "numpy.f2py",
            str(fortran_src),
            "-m",
            "pypns._lmder",
            "-h",
            str(generated_c),
            "--overwrite-signature",
        ]
        subprocess.check_call(cmd)

        # Tell setuptools to compile the generated C source
        ext.sources = [str(generated_c)]
        ext.include_dirs.append(numpy.get_include())

        super().build_extension(ext)


ext_modules = [
    Extension(
        name="pypns._lmder",
        sources=[],  # Will be filled by f2py
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": F2PyBuildExt},
)
