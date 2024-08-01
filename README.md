# Eoles

The Eoles model optimizes the investment and operation of the energy system in order to minimize the total cost while satisfying exogenous energy demands. \
Here is a presentation of an earlier version of the model: _http://www.centre-cired.fr/quel-mix-electrique-optimal-en-france-en-2050/_ \
Most of the model versions, as well as published articles using them, are presented at https://www.centre-cired.fr/the-eoles-family-of-models/

---

### Launching the model with Pyomo

---

#### **Installing the dependencies**

The model requires some python packages as well as the Gurobi solver. 
We provide both a requirements.txt and an environment.yml file for the packages dependencies, and a guide to getting an academic licence for Gurobi.

* **Python** :
Python is an interpreted programming language, with Pyomo it will allow us to model Eoles. \
You can download the latest version on the dedicated website : *https://www.python.org/downloads/* \
Then you just have to install it on your computer. \
If you plan to install Conda or if you have Conda installed on your computer,
Python is likely to be already installed.
The model requires python 3 to run properly, and it is recommended to use at least python 3.10 (older versions might work but were not tested)

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

* Installing dependencies with **Conda**:
Navigate to your directory of choice.
Create the environment and install dependencies : ```conda env create -f environment.yml```
Activate the environment : ```conda activate envEOLES```
If you wish to use Jupyter Notebook :
use ```conda install -c anaconda ipykernel``` and ```python -m ipykernel install --user --name=envEOLES```
The environment will then appear in the kernel list.

* Installing dependencies with **Pip**:
Navigate to your directory of choice.
Create a virtual environment: ```python -m venv envEOLES```
If you use another name for the environment and plan on pushing changes to the github, remember to exclude the environment folder from the commit.
Activate the virtual environment :
Windows : ```envEOLES\Scripts\activate```
macOS/Linux: ```source envEOLES/bin/activate```
Install dependencies : ```pip install -r requirements.txt```

* **Solver** :
The solver that this model uses is Gurobi. \
This solver is commercial but free licenses are available for academics. \
This solver has the advantage to be much faster than other open-source solvers, such as cbc.
More information about Gurobi here : _https://www.gurobi.com/_ \
You can also use another solver if you wish. \
To use Gurobi :
    * Create an account and download Gurobi Optimizer here : _https://www.gurobi.com/downloads/_
    * Request a free academic license here : _https://www.gurobi.com/downloads/end-user-license-agreement-academic/_
    * Use the ```grbgetkey``` command to install your license. The exact command is given with your license.
To use Gurobi on Inari :
	* Request another free academic licence and use the ```grbgetkey``` command to install it on your Inari user account
	* Add the following lines to your .bashrc :
		```# gurobi path and licence
		export GUROBI_HOME=/data/software/gurobi1002/linux64
		export PATH=$GUROBI_HOME/bin:$PATH
		export GRB_LICENSE_FILE=/home/hippolyte/gurobi.lic```




#### **How to get the code :**

If you don't have git installed on your computer, you can download the folder on Github or Zenodo. \
Else, you can retrieve the files from this Github through the command line : \
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

---

This README was originally written by Quentin Bustarret
