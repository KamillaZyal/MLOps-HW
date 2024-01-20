# MLOps-HW
 Homework for the MLOps course
# 📚 Project description
This project uses [pytorch-lightning](https://lightning.ai/) to do digit recognition on the MNIST dataset.
### About Dataset
_The MNIST dataset provided in CSV format_

This dataset uses the work of [Joseph Redmon](https://pjreddie.com/) to provide the [MNIST dataset in a CSV format](https://pjreddie.com/projects/mnist-in-csv/).

The dataset consists of two files:
1. `mnist_train.csv`
2. `mnist_test.csv`

The `mnist_train.csv` file contains training examples and labels. The `mnist_test.csv` contains test examples and labels. Each row consists of 785 values: the first value is the label (a number from 0 to 9) and the remaining 784 values are the pixel values (a number from 0 to 255).
## ▶ Running Model
**If you are not using Windows, change .bat to the appropriate file extension.**
- train model: `python train.py` 
- predict model: `python infer.py`
- run Mlflow server: `python run_server.py`

*or use*
```
.\run_model.bat
.\run_server.bat
```
**Run Mlflow Server**
```
mlflow server --host 127.0.0.1 --port 5000  
```
## 📁 Structure
```bash
  │                                     
  ├───.dvc                                 # DVC installation directory
  │       .gitignore
  │       config
  ├───configs                              # configuretion files
  │       config.yaml
  ├───data                                 # data files using dvc
  │   ├───test                             # test dataset
  │   │       .gitignore
  │   │       mnist_test.csv.dvc          
  │   ├───examples                         # examples for server testing 
  │   │       .gitignore
  │   │       mnist_examples.csv.dvc  
  │   └───train                            # train dataset
  │           .gitignore
  │           mnist_train.csv.dvc
  ├───mnist                                # project files
  │   │   infer.py                         # script .py to predict the model
  │   │   train.py                         # script .py to train and save the model
  │   ├───datasets                         # files for working with datasets
  │   │       dataset.py                   # module to load the dataset 
  │   ├───models                           # model files                  
  │   │       model.py                     
  │   └───utils                            # utilities 
  │           utils.py                     # different utils functions
  ├───tracking-server                      # files for mlflow server with nginx + docker 
  |   |   ...
  ├─── .dvcignore                          # marks which files and/or directories should be excluded when traversing a DVC project.
  ├─── .gitignore                          # marks Git which files and/or directories to ignore when committing your project to the GitHub repository
  ├─── run_model.bat                       # file .bat for run model (training+inference)
  ├─── run_server.bat                      # file .bat for run mlflow server
  ├─── infer.py                            # run mlflow server using python
  ├─── train.py                            # run training model using python
  ├─── infer.py                            # run inference model using python
  ├─── .pre-commit-config.yaml             # identifying simple issues before submission to code review
  ├─── pyproject.toml                      # configuretion file .toml for poetry
  └─── README.md                           # the top-level README for developers using this project
```
## 🛠️ Installation Steps
#### For development
- Step 01: Make directory for project and `cd <name_new_dir>`
- Step 02: Clone the repo in your local system git clone <url> and check the current directory (run `cd MLOps-HW` if necessary)
- Step 03: Create and activate virtaul environment (for example, using `virtualenv`)
- Step 04: Run `poetry intsall`
- Step 05: Run `pre-commint install` and `pre-commit run -a`
- Step 06: Run Mlflow server (`mlflow server --host 127.0.0.1 --port 5000`)
## 🖥️ CLI
#### Initial example for Windows
```
git clone https://github.com/KamillaZyal/MLOps-HW.git
cd MLOps-HW
virtualenv .
Scripts\activate
poetry install
pre-commit install
pre-commit run -a
mlflow server --host 127.0.0.1 --port 5000
.\run_model.bat
.\run_server.bat
```
