import subprocess

import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict

"""script to ensure all required pip and apt packages are installed
    author Jochen Zoellner
    """

if __name__ == "__main__":
    # dependencies can be any iterable with strings,
    # e.g. file line-by-line iterator
    dependencies = [
        "pip>=8.1",
        "pandas",
        "scipy",
        "Pillow",
        "scikit-image",
        "numpy==1.17.4",
        "shapely",
        "descartes",
        "PyPDF2",
    ]

    # here, if a dependency is not met, a DistributionNotFound or VersionConflict
    # exception is thrown.
    print("checking requirements...")
    to_install_list = []
    for package in dependencies:
        try:
            pkg_resources.require([package])
            print("{} is installed".format(package))
        except (VersionConflict, DistributionNotFound) as ex:
            print("{} is not installed".format(package))
            to_install_list.append(package)

    if len(to_install_list) > 0:
        print("\nThe following packages are missing: {}".format(" ".join(to_install_list)))

        answer = input("Do you wish to install these packages? (YES/no)")
        print(answer)
        if answer in ["yes", "YES", "Yes", "y", "Y", ""]:
            print("installing...")
            for module in to_install_list:
                print("install...: {}".format(module))
                print("run in subprocess: {}".format(" ".join(["pip", "install", "--upgrade", module])))
                subprocess.call(["pip", "install", "--upgrade", module])
            print("All requiered PIP packages are installed now. Have fun!")
        else:
            print("Aborted.")
            exit(1)
    else:
        print("All requiered PIP packages are already installed. Have fun!")

    # if int(sys.version[0]) < 3:
    #     logging.warn("apt-package check is only supported for python 3. Pleas check if 'libsox-fmt-mp3' "
    #           "is installed by running:\n sudo apt install libsox-fmt-mp3")
    # else:
    #     import apt
    #     cache = apt.Cache()
    #
    #     apt_packages = ["libsox-fmt-mp3"]  # add additional requiered apt-packages/programs here
    #     missing_modules = []
    #     for apt_module in apt_packages:
    #         if apt_module in cache and cache[apt_module].is_installed:
    #             print("{} is installed.".format(apt_module))
    #         else:
    #             print("{} is missing.".format(apt_module))
    #             missing_modules.append(apt_module)
    #
    #     if len(missing_modules) > 0:
    #         print("You should install the missing module(s) via apt by runnig the following in a "
    #               "terminal:\nsudo apt install {}".format(" ".join(missing_modules)))
    #     else:
    #         print("All requiered APT packages are already installed. Have fun!")

    print("pleas check apt-package installation with: 'sudo apt install python3-tk' ")

    print("Finished successfully.")
