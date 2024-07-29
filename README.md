# Eoles

Eoles model performs optimization of the investment and operation of the energy system in order to minimize the total cost while satisfying energy demand. \
Here is a presentation of an earlier version of the model: _http://www.centre-cired.fr/quel-mix-electrique-optimal-en-france-en-2050/_ \

---

### Launching the model with Pyomo

---

#### **Installing the dependencies**

In order to run the model you will need to install some dependencies that this program needs to run:

* **Python** :
Python is an interpreted programming language, with Pyomo it will allow us to model Eoles. \
You can download the latest version on the dedicated website : *https://www.python.org/downloads/* \
Then you just have to install it on your computer. \
If you plan to install Conda or if you have Conda installed on your computer,
Python is likely to be already installed.
The model requires python 3 to run properly.

* **Conda** ou **Pip** depending on your preference:
Conda and Pip are package managers for Python.
    * **Conda** \
    You can find all the information you need to download and install Conda here:  \
    _https://docs.conda.io/projects/conda/en/latest/user-guide/install/_ \
    __Be careful to choose the version of conda according to the version of Python !!!__ \
    You can install Miniconda which is a minimal version of conda, \
  this allows you to not install all the packages included in conda,
  but you can install only those that you want.
    * **Pip** \
    You can find all the necessary information to download and install pip here: \
    _https://pip.pypa.io/en/stable/installing/_ \
    Pip is installed by default with Conda.

* **Pandas** :
Pandas is a Python library that allows you to manipulate and analyze data easily. \
Pandas is open-source. \
Open a command line and type this to install pandas: \
```conda install pandas```, with Conda \
```pip install pandas```, with Pip \
You can find more information here: _https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html_

* **Pyomo** :
Pyomo is an optimization modeling language based on the Python language. \
Pyomo is open-source.\
Open a command interface and type this to install pyomo : \
```conda install -c conda-forge pyomo```, with Conda \
```pip install pyomo```, with Pip \
You can find more information here: _https://pyomo.readthedocs.io/en/stable/installation.html_

* **Solver** :
The solver that this model uses is Gurobi. \
This solver is commercial but free licenses are available for academics. \
This solver has the advantage to be much faster than other open-source solvers, such as cbc.
More information about Gurobi here : _https://www.gurobi.com/_ \
You can also use another solver if you wish. \
To use Gurobi :
    * Create an account and download Gurobi Optimizer here : _https://www.gurobi.com/downloads/_
    * Request a free academic license here : _https://www.gurobi.com/downloads/end-user-license-agreement-academic/_
    * Use the ```grbgetkey``` command to import your license. The exact command is given with your license.


#### **How to get the code :**

If you don't have git installed on your computer, you can download the folder on this GitLab. \
Else, you can retrieve the files from this GitLab through the command line : \
```git clone https://github.com/CIRED/EOLES.git``` \
A folder will be created in the current directory with all the files contained in this GitLab.

---

### Using the model

---

The EOLES model is written as a class contained in the ```modelEoles.py``` package. \
Several utility functions (initialization and plotting functions) that could be of interest to the user are in the ```utils.py``` package.\
To use the model, import the ModelEOLES class, create an instance of the model with the desired configuration file then use the different methods to build and solve the model and to extract results. An example ```.py``` file is provided.

---

### Input data

---

Example input data are provided in the **inputs** folder.\
The path to each data file can be modified in the config files.\
Supplementary demand and production profiles can be found in [this Zenodo repository](https://doi.org/10.5281/zenodo.13124746)

The expected data format is clarified for each data type (constant or profile) in the relevant utility function in ```utils.py```.



This README was originally written by Quentin Bustarret
