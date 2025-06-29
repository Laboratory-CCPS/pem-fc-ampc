import os
import subprocess
import logging
import sys

from setuptools import setup
from setuptools.command.install import install
from importlib.util import find_spec

logger = logging.getLogger("setup")
logger.setLevel(logging.INFO) 
logging.basicConfig(format="%(levelname)s: %(message)s")


class CustomInstallCommand(install):
    """Custom install command to handle acados installation."""
    def run(self) -> None:
        with_acados = os.getenv("WITH_ACADOS", "0") == "1"
        add_paths = os.getenv("ADD_PATHS", "0") == "1"

        # install acados
        if with_acados:
            self._install_acados(add_paths)
        super().run()

    @staticmethod
    def _install_acados(add_paths: bool) -> None:
        if find_spec("acados_template") is not None:
            logger.info(f"acados_template version is already installed. Skipping build.")
            return

        logger.info("acados_template is not installed. Proceeding with installation...")
        acados_dir = os.path.abspath("acados")
        try:
            if not os.path.exists(acados_dir):
                logger.debug("Cloning acados repository...")
                subprocess.check_call(["git", "clone", "https://github.com/acados/acados.git"])
                os.chdir(acados_dir)

                logger.debug("Initializing submodules...")
                subprocess.check_call(["git", "submodule", "update", "--recursive", "--init"])

                os.makedirs("build", exist_ok=True)
                os.chdir("build")

                # Run CMake and make
                logger.debug("Building acados...")
                subprocess.check_call([
                    "cmake",
                    "-DACADOS_WITH_QPOASES=ON",
                    "-DACADOS_WITH_OSQP=ON",
                    "-DACADOS_WITH_DAQP=ON",
                    "-DACADOS_SILENT=ON",
                    ".."
                ])
                subprocess.check_call(["make", "install", "-j4"])
                os.chdir("../..")
            else:
                logger.debug("acados repository already exists. Skipping clone and build.")

            logger.debug("Installing acados_template...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "acados/interfaces/acados_template"])

            if add_paths:
                logger.debug("Adding paths to bashrc...")
                acados_root = os.path.abspath("acados")
                bashrc_path = os.path.expanduser("~/.bashrc")
                with open(bashrc_path, "a") as bashrc:
                    bashrc.write(f'\nexport LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"{acados_root}/lib"\n')
                    bashrc.write(f'export ACADOS_SOURCE_DIR="{acados_root}"\n')
            
        except Exception:
            raise RuntimeError(f"Error during acados installation. Please follow the manual installation steps from https://docs.acados.org/installation")


setup(
    cmdclass={
        "install": CustomInstallCommand
    },
)
