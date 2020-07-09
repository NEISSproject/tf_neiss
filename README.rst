# tf_neiss

[![Tagged Release](https://img.shields.io/badge/release-v0-blue.svg?longCache=true)](CHANGELOG.md)
[![Development Status](https://img.shields.io/badge/status-planning-lightgrey.svg?longCache=true)](ROADMAP.md)
[![Build Status](https://img.shields.io/badge/build-unknown-lightgrey.svg?longCache=true)](https://travis-ci.org)
[![Build Status](https://img.shields.io/badge/build-pending-lightgrey.svg?longCache=true)](https://www.appveyor.com)
[![Build Coverage](https://img.shields.io/badge/coverage-0%25-lightgrey.svg?longCache=true)](https://codecov.io)

Tensorflow toolkit for NEISS collaboration and partners - learning framework based on tensorflow 2.x

_**Note:** This project was initially created with the cookiecutter template [dough](https://gitlab.com/dboe/dough)!_ :cookie:

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [License](#license)

## Features

## Requirements

## Installation

## Usage
### File structure
proposed project strcture:
- ~/devel/projects/<my_current_project>/tf_neiss
- ~/devel/projects/<my_current_project>/data
- ~/devel/projects/<my_current_project>/lists
- ~/devel/projects/<my_current_project>/models

where \~/devel/projects/<my_current_project> is the python working directory.
and PYTHONPATH=~/devel/projects/<my_current_project>/tf_neiss

### Virtual python environment
- make sure python3 packages are installed:
~~~~{.bash}
sudo apt-get install python3-pip python3-dev python-virtualenv
~~~~
- create new virtualenv in ~/envs/p3_tf:
~~~~{.bash}
virtualenv --no-site-packages -p python3 ~/envs/p3_tf
~~~~
- activate virtualenv
~~~~{.bash}
source ~/envs/p3_tf/bin/activate
~~~~
use *deactivate* to disable virtualenv for current shell

WITHIN the virtualenv install required pip-packages:
~~~~{.bash}
python tf_neiss/tests/test_requirements.py
~~~~
Maybe: Install tensorflow 2.x without gpu-support via pip:

~~~~{.bash}
pip install tensorflow tensorflow-addons
~~~~
For gpu support you need a tensorflow-gpu package compiled with the right cuda and cudnn versions. There is a compiled tensorflow on the neiss-gpu-server which can be used for it.
You need tensorflow-addons==0.6.0 for tensorflow==2.0 or tensorflow_addons==0.7.0 for tensorflow==2.1

~~~~{.bash}
pip install /mnt/data/share/.../<package-name>.whl
~~~~

### Start provided tests
WITHIN the virtualenv:
~~~~{.bash}
cd tf_neiss/tests/workdir_triangel2d
sh test_triangle2d_full.sh
~~~~
If everything is set up correct, it will generate a few train data, train a model and load the model for evaluation (takes only a few miniutes)

As a result there is a pdf-file with triangles and their predictions in the upper plot. You should see first attemps of a reconsturction (far from good).
- comment the Clean Up Part to inspect intermediate results like model or tensorboard logs.
- This test is an example which helps you to integrade your own model in this framework.
- Only files used in the test are up to date for now. There are some function not used yet or deprecated, just ignore them.

#### Test with gpu
~~~~{.bash}
sh test_triangle2d_gpu_full.sh
~~~~
to run the test with gpu device 0.



## Development

## License

See [LICENSE](LICENSE)
