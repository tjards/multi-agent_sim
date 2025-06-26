#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 19:45:47 2024

This program implements Continuous Action Learning Automata (CALA) to learn
    an arbitrary set of actions for n-states

@author: tjards

"""

#%% Import stuff
# ---------------
import numpy as np
import matplotlib.pyplot as plt
import config.configs_tools as configs_tools
from planner.techniques.utils import quaternions as quat
config_path=configs_tools.config_path

#%% Simulations parameters
# ---------------------

actions_range = 'angular'       # 'linear', 'angular' (impacts clipping)

#action_min      = 0             # minimum of action space
#action_max      = 2*np.pi     # maximum of action space

action_min      = -np.pi             # minimum of action space
action_max      = np.pi     # maximum of action space

#%% Hyperparameters
# -----------------
learning_rate   = 0.001     # rate at which policy updates
variance        = 0.2       # initial variance
variance_ratio  = 10        # default 1, permits faster/slower variance updates
variance_min    = 0.1     # default 0.001, makes sure variance doesn't go too low
epsilon         = 1e-8

counter_max = 10             # when to stop accumualating experience in a trial
reward_mode = 'target'       # 'target' = change orientation of swarm to track target 

#%% Learning Class
# ----------------
class CALA:
    
    # initialize
    def __init__(self, num_states):
        
        # load parameters into class
        self.num_states     = num_states
        self.action_min     = action_min
        self.action_max     = action_max
        self.learning_rate  = learning_rate
        self.means          = np.random.uniform(action_min, action_max, num_states) #means
        self.variances      = np.full(num_states, variance) #variances
        
        # Dirichlet distribution with a = 1 , 
        #    inject non-uniform influence or bias across states 
        self.asymmetry  = np.random.rand(num_states)
        self.asymmetry  /= np.sum(self.asymmetry)  # Normalize to sum to 1
        
        # initialize actions 
        self.action_set = 0*np.ones((num_states))

        # counter        
        self.counter_max    = counter_max 
        self.counter        = np.random.uniform(0, self.counter_max, num_states).astype(int) - 500 # all agents start at differnt places
        #self.counter        = np.zeros(num_states) - 750# all in synch now, but do asynch (above) later

        # store environment variables throughout the trial
        self.reward_mode      = reward_mode
        self.environment_vars = np.zeros(num_states)        

        # store stuff
        self.mean_history       = []
        self.variance_history   = []
        self.reward_history     = []
        
        # store the configs
        configs_tools.update_configs('CALA', [
            ('num_states', num_states),
            ('action_min', action_min),
            ('action_max', action_max),
            ('learning_rate', learning_rate),
            ('reward_mode', reward_mode)
        ] )

    
    def compute_multi_reward(self, target, centroid, focal):
        
        approach = 'turret'  # turret 
        
        if self.reward_mode == 'target':
                
            # use origin if no target provided
            if target.shape[1] == 0:
                target_vec = -centroid[0:3, 0]
            else:
                target_vec = target[0:3, 0] - centroid[0:3, 0]
        
            # current 
            focal_vec = focal[0:3] - centroid[0:3,0]
        
            # Normalize
            target_vec /= (np.linalg.norm(target_vec) + epsilon)
            focal_vec /= (np.linalg.norm(focal_vec) + epsilon)
            
            if approach == 'turret':
            
                # get cosine similarity 
                multi_reward = (np.dot(focal_vec, target_vec) + 1) / 2
                            
        else:
            
            multi_reward = 0.0
            
        return multi_reward
        
        

    # main lemniscate learning
    def learn_lemni(self, state, state_array, centroid, focal, target, neighbours, mode, allow_ext_reward, ext_reward = 0):
  
        
        # if multiple rewards to consider
        if allow_ext_reward:
            
            reward = ext_reward
            
        # if not, update yourself    
        else:
  
            reward = self.update_reward_increment(state, state_array, centroid, focal, target, mode)
        
        self.step(state, reward)
    
        if neighbours is not None and len(neighbours) > 1:
            idx = list(neighbours).index(state)
            lag = neighbours[(idx - 1) % len(neighbours)]
            lead = neighbours[(idx + 1) % len(neighbours)]
    
            self.share_statistics(state, [lag, lead], 'actions')
            self.share_statistics(state, [lag, lead], 'rewards')


    # seek consensus between neighbouring rewards (state, list[neighbours])
    def share_statistics(self, state, neighbours, which):
        
        alpha = 0.9                             # for symetric sharing 
        alpha_asym = self.asymmetry[state]      # for asymetric sharing
        
        for neighbour in neighbours:
            
            # if shareing rewards
            if which == 'rewards':
        
                self.means[state]       = alpha * self.means[state] + (1-alpha)*self.means[neighbour]
                self.variances[state]   = alpha * self.variances[state] + (1-alpha)*self.variances[neighbour]
            
            # if sharing actions (i.e., force a common distribution indirectly through action-level consensus)
            elif which  == 'actions':
                
                self.action_set[state] = alpha_asym * self.action_set[state] + (1-alpha_asym)*self.action_set[neighbour]
           
    # select action
    def select_action(self, state):
        
        # pull mean and variance for given state
        mean        = self.means[state]
        variance    = self.variances[state]
        
        # select action from normal distribution
        action = np.random.normal(mean, np.sqrt(variance))
        
        # return the action, onstrained using clip()
        if actions_range == 'linear':
        
            return np.clip(action, self.action_min, self.action_max)
        
        elif actions_range == 'angular':
            
            #return np.mod(action, 2 * np.pi)
            return (action + np.pi) % (2 * np.pi) - np.pi
    
    # update policy 
    def update_policy(self, state, action, reward):
 
        # pull mean and variance for given state
        mean        = self.means[state]
        variance    = self.variances[state]
        
        # update mean and variance based on reward signal
        self.means[state]       += self.learning_rate * reward * (action - mean)
        self.variances[state]   += variance_ratio * self.learning_rate * reward * ((action - mean) ** 2 - variance)
        
        # constrain the variance 
        self.variances[state] = max(variance_min, self.variances[state])

    # ****************************
    # ASYCHRONOUS EXTERNAL UPDATES
    # ****************************


    def update_reward_increment(self, k_node, state, centroid, focal, target, mode):
        
      
        if self.reward_mode == 'target':
            
            reference = 'local'
            
            if reference == 'local':
            
                # get action for this mode
                action = self.action_set[k_node]
        
                reward = 0
                print('need to add local frames')
        
        return reward
        

    # note: this is very messy, clean up object-oriented approach
    def step(self, state, reward):
  
        # increment counter
        self.counter[state] += 1

        # check if update threshold reached
        if self.counter[state] >= self.counter_max:
      
            # update the policy
            self.update_policy(state, self.action_set[state], reward)
            # reset counter
            self.counter[state] = 0  
            #select a new action
            self.action_set[state] = self.select_action(state)

        # log history
        self._log_state(state, self.action_set[state], reward)

    # store current step info into history buffers
    def _log_state(self, state, action, reward):
    
        # expand storage if necessary
        if len(self.mean_history) <= state:
            self.mean_history.append([])
            self.variance_history.append([])
            self.reward_history.append([])

        # store data for this state
        self.mean_history[state].append(self.means[state])
        self.variance_history[state].append(self.variances[state])
        self.reward_history[state].append(reward)
    
    
    # ************** #
    #   PLOTS        #
    # ************** #
    
    # plot results
    def plots_set(self):
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
        time_steps = len(self.mean_history[0])
        self.state_colors = []  # reset in case this is run fresh
    
        for state in range(self.num_states):
            mean_array = np.array(self.mean_history[state])
            variance_array = np.array(self.variance_history[state])
            reward_array = np.array(self.reward_history[state])
            std_devs = np.sqrt(variance_array)
    
            # plot the line first to capture color
            line, = axs[0].plot(range(time_steps), mean_array, label=f"state {state}")
            color = line.get_color()
            self.state_colors.append(color)
    
            # correct color applied to the shaded region
            axs[0].fill_between(range(time_steps),
                                mean_array - std_devs,
                                mean_array + std_devs,
                                color=color,
                                alpha=0.3)
    
            axs[1].plot(range(time_steps), variance_array, label=f"state {state}", color=color)
            axs[2].plot(range(time_steps), reward_array, label=f"state {state}", color=color)
    
        axs[0].set_title('Mean with Std Dev')
        axs[1].set_title('Variance')
        axs[2].set_title('Reward')
    
        for ax in axs:
            ax.set_xlabel("Steps")
            ax.legend()
    
        plt.tight_layout()
        plt.show()

    # plot the distributions 
    def plot_distributions_over_time_set(self, steps_to_plot=[0, 10, 25, 50]):
        from scipy.stats import norm
        x = np.linspace(self.action_min - 0.5, self.action_max + 0.5, 500)
    
        # Compute a shared y-axis limit (max PDF value over all states and selected steps)
        y_max = 0
        for state in range(self.num_states):
            for step in steps_to_plot:
                mu = self.mean_history[state][step]
                sigma = np.sqrt(self.variance_history[state][step])
                if sigma > 0:
                    y = norm.pdf(mu, mu, sigma)
                    y_max = max(y_max, y)
        y_max *= 1.1  # add some headroom
    
        # create the figure
        fig, axs = plt.subplots(self.num_states, 1, figsize=(10, 3 * self.num_states))
    
        for state in range(self.num_states):
            ax = axs[state] if self.num_states > 1 else axs
            color = self.state_colors[state] if hasattr(self, 'state_colors') else None
            for idx, step in enumerate(steps_to_plot):
                mu = self.mean_history[state][step]
                var = self.variance_history[state][step]
                sigma = np.sqrt(var)
                y = norm.pdf(x, mu, sigma)
                alpha = 0.1 + 0.8 * (idx / (len(steps_to_plot) - 1))
                ax.fill_between(x, y, color=color, alpha=alpha, label=f"step {step}" if idx == len(steps_to_plot)-1 else None)
    
            ax.set_title(f"State {state} Distribution Evolution")
            ax.set_xlim(self.action_min - 0.5, self.action_max + 0.5)
            ax.set_ylim(0, y_max)  # ← key fix to match the animation
    
        plt.tight_layout()
        plt.show()
        
    # animate the distributions
    def animate_distributions_set(self, interval=50, save_path=None):
        import matplotlib.animation as animation
        from scipy.stats import norm
    
        time_steps = len(self.mean_history[0])
        x = np.linspace(self.action_min - 0.5, self.action_max + 0.5, 500)
    
        # Find global max y for PDF
        y_max = 0
        for state in range(self.num_states):
            for t in range(time_steps):
                sigma = np.sqrt(self.variance_history[state][t])
                mu = self.mean_history[state][t]
                y_max = max(y_max, norm.pdf(mu, mu, sigma))
        y_max *= 1.1

        fig, axs = plt.subplots(self.num_states, 1, figsize=(10, 3 * self.num_states))
        plt.subplots_adjust(hspace=0.4)
        if self.num_states == 1: axs = [axs]
    
        lines, fills = [], []
        for state in range(self.num_states):
            color = self.state_colors[state] if hasattr(self, 'state_colors') else None
            ax = axs[state]
            line, = ax.plot([], [], color=color)
            fill = ax.fill_between(x, np.zeros_like(x), np.zeros_like(x), color=color, alpha=0.3)
            lines.append(line)
            fills.append(fill)
            ax.set_xlim(self.action_min - 0.5, self.action_max + 0.5)
            #ax.set_ylim(0, y_max)
            ax.set_ylim(0, 1)
            ax.set_title(f"State {state} - Action PDF over Time")
    
        time_text = axs[0].text(0.95, 0.95, '', transform=axs[0].transAxes,
                                ha='right', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
        def init():
            for line in lines:
                line.set_data([], [])
            time_text.set_text('')
            return lines + [time_text]
    
        def update(frame):
            for state in range(self.num_states):
                mu = self.mean_history[state][frame]
                sigma = np.sqrt(self.variance_history[state][frame])
                y = norm.pdf(x, mu, sigma)
                lines[state].set_data(x, y)
                #fills[state].remove()
                #fills[state] = axs[state].fill_between(x, y, color=self.state_colors[state], alpha=0.3)
                
                # remove previous collection from axis
                for artist in axs[state].collections:
                    artist.remove()

                # add new one
                fills[state] = axs[state].fill_between(x, y, color=self.state_colors[state], alpha=0.3)
                
            time_text.set_text(f"Time step: {frame}/{time_steps}")
            return lines + fills + [time_text]
    
        ani = animation.FuncAnimation(fig, update, frames=time_steps,
                                      init_func=init, blit=False, interval=interval)
    
        if save_path:
            ani.save(save_path, writer='pillow', fps=1000 // interval)
        else:
            plt.tight_layout()
            plt.show()
    
        return ani

    # plots
    def all_plots_set(self):
        
        self.plots_set()
        self.plot_distributions_over_time_set()
        #anim = self.animate_distributions_set(interval=50, save_path='RL_animation.gif')

    '''def plot_reward_surface_set(self, reward_fn, xlim=(-3.5, 3.5), num_points=200):
        """
        Plot reward surfaces for all states, using latest mean and variance.
        """
        import matplotlib.pyplot as plt
        import numpy as np
    
        n_states = len(self.mean_history[-1])
        fig, axs = plt.subplots(n_states, 1, figsize=(8, 1.8 * n_states), sharex=True)
    
        x = np.linspace(xlim[0], xlim[1], num_points)
    
        for i in range(n_states):
            mu = self.mean_history[-1][i]
            sigma = np.sqrt(self.variance_history[-1][i])
    
            rewards = []
            for xi in x:
                state = np.zeros(n_states)
                state[i] = xi
                rewards.append(reward_fn(state))
    
            axs[i].plot(x, rewards, label='Reward')
            axs[i].axvline(mu, color='red', linestyle='--', label='Mean')
            axs[i].fill_between(
                x, 0, max(rewards),
                where=((x > mu - sigma) & (x < mu + sigma)),
                color='red', alpha=0.1, label='1σ'
            )
            axs[i].set_title(f'State {i} Reward Surface')
            axs[i].grid(True)
            axs[i].legend()
    
        plt.tight_layout()
        plt.show()'''


# manual calls
# ------------
# Controller.Learners['lemni_CALA'].animate_distributions_set()
# Controller.Learners['lemni_CALA'].all_plots_set()
#Controller.Learners['lemni_CALA'].plot_reward_surface_set(reward_fn=compute_reward)

