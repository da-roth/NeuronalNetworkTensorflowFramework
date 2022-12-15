# MonteCarloLearning

The goal of this repo is to introduce deep learning with sampled data and to create a flexible framework for its application.s

The idea to study the different approaches stems from
https://arxiv.org/abs/2102.08734v1

The idea and the first draft of the programming code stems from
https://github.com/differential-machine-learning/notebooks

### Installation
(Python used: Python 3.10.8)

# 1. Open command prompt and naviagte to folder (e.g. C\dev\MonteCarloLearning) 
# 2. Create virtual environment
python -m venv .venvNN
# 3. Activate virtual environment in Command Prompt:
.venvNN\Scripts\activate.bat
# 4. Install packages from requirements.txt, 
pip install -r requirements.txt


### To add new packages: Update/Generate requirements.txt from current installation
pip freeze > requirements.txt


### Useful commands:

# (Use direct path of Python if other python version is required, e.g. C\Users\AppData\Local\Programs\Python\Python39\python.exe)
# Print used Python (prints version of global used or Venv if in a Venv)
python -V
# Deactivate environment
deactivate
# Python package lookup path
echo %PATH%
# pip upgrade can be upgraded by
c:\dev\test\.venvNN\Scripts\python.exe -m pip install --upgrade pip
# or if Veng is activated just
python -m pip install --upgrade pip
