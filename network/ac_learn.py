from tqdm import tqdm #progress bar
import numpy as np
import pandas as pd

import nengo
import pytry
import gym
import learnrules as rules
import representations as rp
import minigrid_wrap
from actor_critic import ActorCritic, ActorCriticLDN

##Softmax Function used for selecting next action
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

class ActorCriticLearn(pytry.Trial):
    def params(self):
        self.param('gym environment: MiniGrid, MountainCar, CartPole or LunarLander', env='MiniGrid')
        self.param('representation', rep = rp.OneHotRep((8,8,4)))
        self.param('number of learning trials', trials=1000),
        self.param('length of each trial', steps=200),
        self.param('select learning rule: rules.ActorCriticTD0, rules.ActorCriticTDn or rules.ActorCriticTDLambda', 
                   rule=rules.ActorCriticTD0)
        self.param('n for TD(n)', n=None),
        self.param('learning rate', alpha=0.1),
        self.param('action discount factor', beta=0.85),
        self.param('value discount factor', gamma=0.95),
        self.param('lambda value', lambd=None),
        self.param('number of neurons', n_neurons=None),
        self.param('neuron sparsity', sparsity=None),
        self.param('generate encoders by sampling state space', sample_encoders=False)
        self.param('choose whether to report the weights', report_weights=False)
        self.param('Set number of dimensions', dims=None)
        
        ## LDN Parameters ##
        self.param('Set whether LDN memories are continuous', continuous=False)
        self.param('Set theta for LDN', theta=0.1)
        self.param('Set q for LDN', q=10)
         
    
    def evaluate(self, param):
        ## Lists for saving data
        Ep_rewards=[]
        Rewards=[]
        Values=[] 
        Roll_mean = []
        
        ## Set environment
        if param.env == 'MiniGrid':
            env = minigrid_wrap.MiniGrid_wrapper()
        elif param.env == 'MountainCar':
            env = gym.make("MountainCar-v0")
        elif param.env == 'CartPole':
            env = gym.make('CartPole-v1')
        elif param.env == 'LunarLander':
            env = gym.make('LunarLander-v2')
            
        ## Set parameters
        rep = param.rep
        trials = param.trials
        steps = param.steps
        rule = param.rule
        n = param.n
        alpha = param.alpha
        beta = param.beta
        gamma = param.gamma
        lambd = param.lambd
        n_neurons = param.n_neurons
        sparsity = param.sparsity
        report_weights = param.report_weights
        n_actions = env.action_space.n
        
        continuous = param.continuous
        theta = param.theta
        q = param.q
        
        ## If using continuous time, need to have the agent wait for 2 time steps in each state
        wait = 2
        
        ## Set sample encoders to True to generate place cells
        if param.sample_encoders == "True" and n_neurons is not None:
            pts = np.random.uniform(0,1,size=(n_neurons,len(env.observation_space.high)))
            pts = pts * (env.observation_space.high-env.observation_space.low)
            pts = pts + env.observation_space.low
            encoders = [rep.map(rep.get_state(x, env)).copy() for x in pts]
        else:
            encoders = nengo.Default
        
        ## Choose the AC network - either with or without LDN memories
        if rule == rules.ActorCriticTDtheta:
            ac = ActorCriticLDN(rep, 
                             rule(n_actions=n_actions, n=n, alpha=alpha, beta=beta, gamma=gamma, lambd=lambd), 
                                theta=theta, q=q, continuous=continuous, n_neurons=n_neurons, 
                                sparsity=sparsity, encoders=encoders)
        else:
            ac = ActorCritic(rep, 
                             rule(n_actions=n_actions, n=n, alpha=alpha, beta=beta, gamma=gamma, lambd=lambd), 
                            n_neurons=n_neurons, sparsity=sparsity, encoders=encoders)
        
        ## LEARNING ##
        for trial in tqdm(range(trials)):
            rs=[] ##reward storage
            vs=[] ##value storage
            env.reset() ##reset environment
            update_state = rep.get_state(env.reset(), env) ##get state
            value, action_logits = ac.step(update_state, 0, 0, reset=True) ##get state and action values
            
            ## For each time step
            for i in range(steps): 
                ## Optional command to render the environment every n trials
                #if trial % (trials/20) == 0:
                #    env.render()
                
                ## Choose and do action
                action_distribution = softmax(action_logits) 
                action = np.random.choice(n_actions, 1, p=action_distribution)
                obs, reward, done, info = env.step(int(action))

                ## Get new state
                current_state = rep.get_state(obs, env)

                ## Update state and action values 
                value, action_logits = ac.step(current_state, action, reward)
                
                ## If using continuous time, need to have the agent wait for 2 time steps in each state
                if continuous == True:
                    for w in range(wait):
                        value, action_logits = ac.step(current_state, action, reward)
                        rs.append(reward)
                        vs.append(value.copy())

                rs.append(reward) ##save reward
                vs.append(value.copy()) ##save state value

                ## When using mutli-step back-ups, once the agent has reached the goal you 
                ## need it to sit there for n time steps and continue doing updates
                if done:
                    if n is not None:
                        for j in range(n):
                            ##Update state and action values 
                            reward = 0
                            value, action_logits= ac.step(current_state, action, reward)
                            
                            ## If using continuous time, need to have the agent wait for 2 time steps in each state
                            if continuous == True:
                                for w in range(wait):
                                    value, action_logits = ac.step(current_state, action, reward)
                                    rs.append(reward)
                                    vs.append(value.copy())

                            vs.append(value.copy()) ##save state value
                            rs.append(reward) ##save reward
                    break                    

            Ep_rewards.append(np.sum(rs)) ##Store average reward for episode
            Rewards.append(rs) ##Store all rewards in episode
            Values.append(vs) ##Store all values in episode  
        #env.close() ##Uncomment if rendering the environment
        

        ## Convert list of rewards per episode to dataframe    
        rewards_over_eps = pd.DataFrame(Ep_rewards)
        ## Calculate a rolling average reward across previous 100 episodes
        Roll_mean.append(rewards_over_eps[rewards_over_eps.columns[0]].rolling(100).mean())
        

        return dict(
            episodes=Ep_rewards,
            rewards = Rewards,
            values = Values,
            roll_mean = Roll_mean,
            policy = ac.get_policy(),
            )
