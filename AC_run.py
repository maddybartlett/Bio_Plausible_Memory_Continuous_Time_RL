## Import dependencies ##
## path
import sys
sys.path.insert(0, './network')
import os

## network
import learnrules as rules
import representations as rp
from ac_learn import ActorCriticLearn

## data handling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## Softmax Function ##
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))


## Function for calulating the ideal value given the rewards recieved in the last trial ##
def value(discount, t, ep_rewards):
    d = discount**t
    return np.convolve(d[::-1], ep_rewards[0][-1])[-len(t):]


## Function for plotting the value of each state and the value of each action in each state ##
def plot_table(args, P):
    plt.figure(figsize=(14,14))
    for i in range(4):
        for j in range(4):
            plt.subplot(4, 4, i*4+j+1)
            plt.imshow(P[:,:,j,i])#, vmin=0, vmax=1)
            plt.colorbar()
            if j == 0:
                plt.ylabel(['value', 'turn left', 'turn right', 'forward'][i])
            if i == 0:
                plt.title(['east', 'south', 'west', 'north'][j])
    plt.show()
    plt.savefig(args['data_dir']+"/state_values.pdf")

    
## Function for plotting the policy ##
def plot_policy(args, P):
    plt.figure(figsize=(14,3.5))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        img = np.zeros((8,8,3))
        for x in range(8):
            for y in range(8):
                img[x,y,:] = softmax(P[x,y,i,1:])
        plt.imshow(img)
        plt.title(['east', 'south', 'west', 'north'][i])
    plt.xlabel('Red:LEFT green:RIGHT blue:FORWARD')
    plt.show()
    plt.savefig(args['data_dir']+"/policy.pdf")

    
## Running the Network ##
def main(args):
    ## lists of representations and rules 
    REPS = {'1H' : rp.OneHotRep((8,8,4)), 'SSP': rp.SSPRep(N=3, D=args['dims'], scale=[0.75,0.75,1.0]), 'Grid' : rp.GridSSPRep(3)}
    RULES = {'TDn': rules.ActorCriticTDn, 'TDtheta': rules.ActorCriticTDtheta}
    
    ## lists for saving data
    ep_results = []
    ep_rewards=[]
    ep_values=[]
    ep_rollmean = []
    policy = []
    
    ## run network
    out = ActorCriticLearn().run(rep = REPS[args['rep']], 
                                 rule = RULES[args['rule']],
                                 alpha = args['alpha'], 
                                 beta = args['beta'], 
                                 gamma = args['gamma'],
                                 n_neurons = args['neurons'], 
                                 sparsity = args['sparsity'], 
                                 dims = args['dims'],
                                 n = args['n'],
                                 continuous = args['continuous'],
                                 theta = (0.001 * (n-1)),
                                 q = args['q'],
                                 verbose = False, 
                                 trials = args['trials'],
                                 data_dir = args['data_dir'])
    
    ## get output data 
    ep_results.append(out["episodes"])
    ep_rewards.append(out["rewards"])
    ep_values.append(out["values"])
    policy.append(out["policy"])
    
    ## Plot things
    print('Plotting rewards')
    plt.figure(figsize = (15,5))
    plt.plot(ep_results[0])
    plt.ylabel('Reward')
    plt.xlabel('Learning Trial')
    plt.show()
    plt.savefig(args['data_dir']+"/ep_rewards.pdf")
    
    print('Plotting ideal vs. actual value for last learning trial')
    T = len(ep_rewards[0][-1])
    t = np.arange(0, int(T))
    
    plt.figure(figsize = (15,5))
    plt.plot(t, value(0.95, t, ep_rewards), label='ideal value')
    plt.plot(ep_values[0][-1], label='actual value')
    plt.legend()
    plt.ylabel('Value')
    plt.xlabel('Time Step')
    plt.show()
    plt.savefig(args['data_dir']+"/ideal_val.pdf")
    
    print('Plotting policy')
    ## Plot state and action values   
    plot_table(args, policy[0])
    ## Plot the policy
    plot_policy(args, policy[0])
    
    ## Finished
    print('Done.')
    
    
## Setting Parameters ##
if __name__ == '__main__':
    directory = input("Enter path for saving the data? ") 
    while not os.path.exists(directory):
        print("Path of the file is Invalid")
        directory = input("Enter path for saving the data? ")
        
    representation = "1H"        
    rule = "TDtheta"
        
    alpha = 0.1
    beta = 0.9
    gamma = 0.95
    
    neurons = 3000
    sparsity = 0.1
    dims = 1
    
    n = 2
    continuous = True #set to false to use discrete time. Only relevant when using TDtheta learning rule
    q = 50
    
    trials = 500
    
    params = {'rep':representation, 'rule': rule, 'alpha':alpha,
             'beta': beta, 'gamma':gamma, 'neurons':neurons,
             'sparsity':sparsity, 'dims':dims, 
             'n': n, 'continuous':continuous, 'q':q,
             'trials':trials, 'data_dir':directory}
    main(params)