## 🛠️ Installation

[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/)
[![Downloads](https://static.pepy.tech/badge/ultralytics)](https://clickpy.clickhouse.com/dashboard/ultralytics)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics/)

To install the `ultralytics` package in developer mode, which allows you to modify the source code directly, ensure you have [Git](https://git-scm.com/) and [Python](https://www.python.org/) 3.9 or later installed on your system. Then, follow these steps:

1.  Clone the `ultralytics` repository to your local machine using Git:

    ```bash
    git clone https://github.com/ultralytics/ultralytics.git
    ```

2.  Navigate to the cloned repository's root directory:

    ```bash
    cd ultralytics
    ```

3.  Install the package in editable mode (`-e`) along with its development dependencies (`[dev]`) using [pip](https://pip.pypa.io/en/stable/):

    ```bash
    pip install -e '.[dev]'
    ```

    This command installs the `ultralytics` package such that changes to the source code are immediately reflected in your environment, ideal for development.
