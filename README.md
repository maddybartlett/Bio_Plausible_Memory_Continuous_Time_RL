# Biologically-Plausible Memory for Continuous-Time Reinforcement Learning

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

Running this command without any changes to the file will result in you first being prompted to provide a directory for saving the data and figures. The AC network will then try to solve the MiniGrid task using the TD(n) learning rule in continuous time with $n = 2$. The network will utilise a One-Hot representation of the state. The preset parameter values are the same as those used in the TD(&theta;) condition in the published experiment. <br>
For the TD(n) experiment, the only changes needed are to set rule = TDn and continuous = False. <br>
Once the network has finished running, plots showing the total reward received in each of the 500 learning trials, the ideal vs. actual value for last learning trial and the policy will be presented on the screen and saved to the directory you specified.

Alternatively, you can step through the process of running the network by opening the AC_Run.ipynb file in Jupyter Notebook.

### Experiment

In order to replicate the experiment reported in Bartlett, Dumont, Furlong & Stewart (2022) you will need to run the network twice -- once for each rule. 

For the TD(&theta;) rule, simply run the network in its current state (i.e. with the current parameter settings).
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

## Network:

All of the scripts required to create and run the network can be found in the './network/' folder. <br>

**ac_learn.py** is a pytry trial script which contains the instructions for running the network <br>

**actor_critic.py** is the script needed to construct both Actor-Critic networks. <br>
* The first, '*ActorCritic*', is the network needed for the classic TD learning rules. The history of rewards and state values are stored in arrays.
* '*ActorCriticLDN*' is the network needed for the novel TD(&theta;) learning rule. It utilises the '*LDN*' class to create Legendre Delay Networks for containing memories of the rewards and state values. <br>

**grid_cells.py** is used to create a population of grid cells for representing the agent's state. *This representation method was not used in these experiments*. <br>

**learnrules.py** contains the 4 Temporal Difference learning rules - TD(0), TD(n), TD($\lambda$) and TD($\theta$). This is the script that is used to perform the TD updates, which are performed by the network's '*rule node*'. <br>

**minigrid_wrap.py** is a wrapper for the Gym MiniGrid environment which allows us to interact with this environment in the same way as we would any other OpenAI Gym environment such as mountaincar or cartpole. It creates an *observation_space* attribute, and alters the observations that are returned when an agent steps through the environment to be the agent's x and y coordinate location, and the direction it's facing. 

**representations.py** provides 4 different methods for representing the agent's state in the network: <br>
* **NormalRep** - normalises the observation values so that they are between -1 and 1
* **OneHotRep** - creates a one-hot representation
* **SSPRep** - utilises Spatial Semantic Pointers (i.e. a vector symbolic architecture) to represent the state
* **GridSSPRep** - constructs a population of grid cells which represent SSPs (see Dumont & Eliasmith, 2020 for further details on this method) <br>

*The experiments reported in this paper only used the One-Hot representation method.* 

## Temporal Difference Learning Rules:

In this work we implemented and compared the performance of two Temporal Difference (TD) learning rules: TD(n) and TD($\theta$). <br>
TD(&theta;) is a novel rule, developed to provide a method of implementing Temporal Difference learning in continuous-time. 

The following sections outline the two rules, providing their mathematical descriptions and highlighting how they have been implemented in code (see the learnrules.py script).

### TD(n):

The TD(n) or the n-step back-up rule is implemented in discrete time. <br> 
With this rule, each timestep involves updating the values of the previous *n* states, using the rewards gained between the timesteps *t* and *t + n*. <br>
To implement this method we first have to calculate the n-step target (or return) for each step. This is done according to the following equation:

G<sub>t:t+n</sub> = R<sub>t+1</sub> + &gamma;R<sub>t+2</sub> + ... + &gamma;<sup>n-1</sup>R<sub>t+n-1</sub> + &gamma;<sup>n</sup> V<sub>t+n-1</sub>(S<sub>t+n</sub>)

Where: <br>
G<sub>t:t+n</sub> = the n-step return for steps t to t+n <br>
R<sub>t+1</sub> = reward gained moving from state(t) to state(t+1) <br>
V<sub>t+n-1</sub>(S<sub>t+n</sub>) = value (before update) of the state the agent is currently in

This is then incorporated into the state-value learning algorithm for updating the value of each state:

V<sub>t+n</sub>(S<sub>t</sub>) = V<sub>t+n-1</sub>(S<sub>t</sub>) + &alpha;(G<sub>t:t+n</sub> - V<sub>t+n-1</sub>(S<sub>t</sub>))

Where: <br>
V<sub>t+n</sub>(S<sub>t</sub>) = the updated value of the state agent was in n-steps ago <br>
V<sub>t+n-1</sub>(S<sub>t</sub>) = the old value of the state agent was in n-steps ago <br>
(G<sub>t:t+n</sub> - V<sub>t+n-1</sub>(S<sub>t</sub>)) = the error term

#### In the code:

In the learnrules.py script, the history of rewards, states, values and actions are stored in lists:

```
state_memory = [] ##list for storing the last n states
value_memory = [] ##list for storing the last n values
reward_memory = [] ##list for storing the last n rewards
action_memory = [] ##list for storing the last n chosen actions
```

These lists are emptied at the start of each learning trial:

```
if reset:
    count = 0
    state_memory.clear()
    value_memory.clear()
    reward_memory.clear()
    action_memory.clear()
```

New information is added on each time step. For example:

```
reward_memory.append(reward)
```

and used for updating the value function. 

At the end of each time-step, we delete the oldest entry in the list. This ensures that the lists only contain values relevant to the update. (The length of the lists is determined by the value of $n$):

```
state_memory.pop(0)
value_memory.pop(0)
reward_memory.pop(0)
action_memory.pop(0)
```

For the TD error term, we need to use the rewards list in order to calculate a discounted sum of the rewards received in the last $n$ timesteps:

R<sub>t+1</sub> + &gamma;R<sub>t+2</sub> + ... + &gamma;<sup>n-1</sup>R<sub>t+n-1</sub>

In the code, we do this by taking all of the values in the rewards list, and multiplying them by the discount term $\gamma$ (gamma):

```
Rs = self.gamma**np.arange(n)*reward_memory[:]
```

Then summing them together in order to then calculate the n-step return (target): 

```
target = np.sum(Rs) + ((self.gamma**n)*current_state_value)
```


### TD(&theta;):

In contrast, TD(&theta;) can be implemented in continuous time. It is possible to use these scripts to implement TD($\theta$) in discrete time, however, the published experiment used the continuous time implementation. 

This novel Temporal Difference learning rule uses Legendre Delay Networks (Voelker, Kajić, & Eliasmith, 2019) to encode the history of events. <br>
Legendre Delay Networks leverage the properties of Legendre polynomials which can be used to represent functions over fixed windows of time. <br>
The LDN is a dynamic system that approximates the Legendre polynomial coefficients of an input signal over a sliding history window of length &theta;. <br>

Recall that with discrete rules like TD(n), the history of rewards, states, values and actions are often stored as explicit lists containing $n$ values (where $n = $ number of time steps in history). 
In contrast, the LDN stores a continuous-time signal over a window of length $\theta$. Thus the RL task no longer needs to be divided into discrete time steps. 

**Note**: in this repository, only the reward history is stored in an LDN.

In terms of the learning rule, this change means that we can't calculate a weighted sum of rewards as we would in TD(n). Instead we calculate the discounted integral over the reward history. 
So whilst the TD(n) error term is:

&delta; = G<sub>t:t+n</sub> - V<sub>t+n-1</sub>(S<sub>t</sub>)

The TD(&theta;) error term is:

&delta; = &int;<sub>0</sub><sup>1</sup> &gamma;<sup>1-&tau;</sup> **R**(t - &theta;&tau;)d&tau; + &gamma;V(t) - V(t-&theta;) <br>
        = (&int;<sub>0</sub><sup>1</sup> &gamma;<sup>1-&tau;</sup> **P**<sup> q<sub>r</sub> </sup>(&tau;)d&tau;)m<sub>R</sub>(t) + &gamma;V(t) - V(t-&theta;)

Where: <br>
**P**<sup> q<sub>r</sub> </sup> = the LDN storing the history of rewards

This operation of taking the integral over the LDN window is handled by the ```reward_decorders``` defined in the ActorCriticLDN. 


## Citation:

Please use this bibtex to reference the paper: 

<pre>
@inproceedings{bartlett2022_TDtheta,
  author = {Bartlett, M., Furlong, P., Dumont, N. S.-Y., & Stewart, T.},
  title = {Biologically-plausible memory for continuous-time reinforcement learning.},
  year = {2022, July},
  booktitle={In-Person MathPsych/ICCM 2022},
  url={mathpsych.org/presentation/894.}
 } 
</pre>
