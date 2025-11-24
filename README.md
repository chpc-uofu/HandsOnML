# Hands-On DL
CHPC Notebooks for Hands-On Deep Learning<br>
by Wim R.M. Cardoen (<a href="https://chpc.utah.edu/">CHPC</a>, University of Utah)<br>

+ <a href="./notebooks/lecture1.ipynb">Lecture 1</a>: Simple Shallow Neural Net (Logistic Regression)
+ <a href="./notebooks/lecture2.ipynb">Lecture 2</a>: General Dense Neural Net

## Installation of the Python packages (Linux)
+ Download the <a href="https://github.com/conda-forge/miniforge">miniforge</a> installation script<br>
  `wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh`
+ Install the miniforge base in the directory *DIR*<br>
  `bash ./Miniforge3-Linux-x86_64.sh -b -p $DIR`
+ Modify the `PATH` variable (assuming your `SHELL` is either `sh`, `bash` or `zsh`)<br>
  - `export PATH=$DIR/bin:$PATH`<br>
  - `echo $(which python)`
+ Install the required packages (requirements.txt are stored in *install/*)<br>
  `pip install -r install/requirements.txt`
+ A `miniforge` <a href="https://lmod.readthedocs.io/en/latest/">lmod</a> file can be found in *install/lmod/*

  
## Check installation from CLI
`python install/check_install.py`
