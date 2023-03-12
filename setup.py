import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension):
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        config = "DEBUG" if debug else "RELEASE"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={config}",
        ]

        if self.compiler.compiler_type == "msvc":
            cmake_args += [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG={extdir}{os.sep}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={extdir}{os.sep}",
            ]

        build_args = []

        if hasattr(self, "parallel") and self.parallel:
            build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )


setup(
    # ext_modules=[CMakeExtension(os.sep.join(["neuropy", "test"]))],
    cmdclass={
        "build_ext": CMakeBuild,
    }
)
