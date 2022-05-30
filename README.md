# Biologically-Plausible Memory for Continuous-Time Reinforcement Learning
Companion repository to ICCM paper "Biologically-Plausible Memory for Continuous-Time Reinforcement Learning"

Contributors: Dr. M. Bartlett, 

Repository to accompany Bartlett, Dumont, Furlong & Stewart (2022) "Biologically-Plausible Memory for Continuous-Time Reinforcement Learning" ICCM Paper (LINK).

## Requirements:

You will need to have Jupyter Notebook installed in order to run these scripts. Recommended to install [Anaconda](https://www.anaconda.com/products/individual). 

* Python 3.5+
* OpenAI Gym
* Gym MiniGrid
* PyTry
* Nengo
* Pandas
* Numpy
* Tqdm
* Scipy
* Math
* Matplotlib
* Sys
* Pickle
* Pathlib

## Scripts and How-To Guide:

Download repository as a zip file or clone repository by clicking the "Code" button on the top right. <br>

### Run the Network

You can run the network from the command line using AC_run.py - in the terminal, navigate to the directory containing these scripts and run the command "python AC_run.py". 

Running this command without any changes to the file will result in you first being prompted to provide a directory for saving the data and figures. The AC network will then try to solve the MiniGrid task using the TD(n) learning rule in continuous time with $n = 2$. The network will utilise a One-Hot representation of the state. The preset parameter values are the same as those used in the TD($\theta$) condition in the published experiment. <br>
For the TD(n) experiment, the only changes needed are to set rule = TDn and continuous = False. <br>
Once the network has finished running, plots showing the total reward received in each of the 500 learning trials, the ideal vs. actual value for last learning trial and the policy will be presented on the screen and saved to the directory you specified.

Alternatively, you can step through the process of running the network by opening the AC_Run.ipynb file in Jupyter Notebook.

### Experiment

In order to replicate the experiment reported in Bartlett, Dumont, Furlong & Stewart (2022) you will need to run the network twice -- once for each rule. 

For the TD($\theta$) rule, simply run the network in its current state (i.e. with the current parameter settings).
When running the network from the command line, we suggest setting the path for saving the data to '.\data\MG_TDtheta',
To run the TD(n) rule, make the following changes to the parameters:

* rule = "TDn" (python file) or rules.ActorCriticTDn (ipynb file)
* continuous = False
* data_dir = '.\data\MG_TDn',

### Analysis

To analyse the data from Jupyter Notebook, open and step-through Analysis_Compare_Rules.ipynb.

This script will produce three plots and save them to a 'figures/' directory. The three plots are:

* Plot of rewards recieved in each learning trial, across all learning trials
* Plot of ideal value vs. actual value of the final learning trial
* Plot of state values

No further analysis was done on these data as the intention was just to establish whether the novel TD($\theta$) rule would produce similar results as the standard TD(n) learning rule. 

## Citation:

Please use this bibtex to reference the paper: 

<pre>
<!-- @inproceedings{bartlett2022_TDtheta,
  author = {},
  title = {},
  year = {},
  booktitle={},
 } -->
</pre>
