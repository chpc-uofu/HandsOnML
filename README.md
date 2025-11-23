# Hands-On DL
CHPC Notebooks for Hands-On Deep Learning<br>
by Wim R.M. Cardoen (CHPC, University of Utah)<br>

+ Lecture 1: Simple Shallow Neural Net (Logistic Regression)
+ Lecture 2: General Dense Neural Net

## Installation of the Python packages (Linux)
+ Download the miniforge installation script<br>
  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
+ Install the miniforge binary in the directory *DIR*<br>
  bash ./Miniforge3-Linux-x86_64.sh -b -p $DIR
+ Modify the PATH variable (assuming SHELL is either sh, bash or zsh<br>
  - export PATH=$DIR/bin:$PATH<br>
  - echo $(which python3)
+ Install the required packages (requirements.txt are stored in *install/*)<br>
  pip install -r install/requirements.txt
  
## Check installation from CLI
python install/check_install.py
