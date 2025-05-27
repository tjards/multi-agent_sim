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
config_path=configs_tools.config_path

#%% Simulations parameters
# ---------------------
#num_states      = 3             # number of states 
action_min      = 0.5 #-1            # minimum of action space
action_max      = 4 # 1             # maximum of action space

# (artificial, just for debugging. would come from env)
#target_actions = np.random.uniform(action_min, action_max, num_states) # randomly assigned target actions

#%% Hyperparameters
# -----------------
learning_rate   = 0.1       # rate at which policy updates
variance        = 0.4       # initial variance
variance_ratio  = 1         # default 1, permits faster/slower variance updates
variance_min    = 0.001     # default 0.001, makes sure variance doesn't go too low

counter_max = 100            # when to stop accumualating experience in a trial
reward_mode = 'centroid'        # 'cmds' = punish commands (start with this), 'centroid' = distance from centroid

# initial means and variances
#means = np.random.uniform(action_min, action_max, num_states)
#variances = np.full(num_states, variance)

#%% Learning Class
# ----------------
class CALA:
    
    # initialize
    #def __init__(self, num_states, action_min, action_max, learning_rate, means, variances):
    def __init__(self, num_states):
        
        # load parameters into class
        self.num_states     = num_states
        self.action_min     = action_min
        self.action_max     = action_max
        self.learning_rate  = learning_rate
        self.means          = np.random.uniform(action_min, action_max, num_states) #means
        self.variances      = np.full(num_states, variance) #variances
        
        # initialize actions 
        self.action_set = np.ones((num_states))

        # counter        
        self.counter_max    = counter_max 
        self.counter        = np.random.uniform(0, self.counter_max, num_states).astype(int) # all agents start at differnt places
        #self.counter        = np.zeros(num_states) # all in synch now, but do asynch (above) later

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


    # select action
    def select_action(self, state):
        
        # pull mean and variance for given state
        mean        = self.means[state]
        variance    = self.variances[state]
        
        # select action from normal distribution
        action = np.random.normal(mean, np.sqrt(variance))
        
        # return the action, onstrained using clip()
        return np.clip(action, self.action_min, self.action_max)
    
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

    def step(self, state, reward):
        """
        Perform asynchronous update for a single state based on internal counter.
        Returns True if update occurred, else False.
        """
        # Select action for the state
        action = self.select_action(state)

        # Increment counter
        self.counter[state] += 1

        # Check if update threshold reached
        if self.counter[state] >= self.counter_max:
            self.update_policy(state, action, reward)
            self.counter[state] = 0  # Reset counter

        # Log history
        self._log_state(state, action, reward)
        
        self.action_set[state] = action

        #return action

    def _log_state(self, state, action, reward):
        """
        Store current step info into history buffers.
        """
        # Expand storage if necessary
        if len(self.mean_history) <= state:
            self.mean_history.append([])
            self.variance_history.append([])
            self.reward_history.append([])

        # Store data for this state
        self.mean_history[state].append(self.means[state])
        self.variance_history[state].append(self.variances[state])
        self.reward_history[state].append(reward)
    
    
    def plots_set(self):
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
        time_steps = len(self.mean_history[0])
        self.state_colors = []  # reset in case this is run fresh
    
        for state in range(self.num_states):
            mean_array = np.array(self.mean_history[state])
            variance_array = np.array(self.variance_history[state])
            reward_array = np.array(self.reward_history[state])
            std_devs = np.sqrt(variance_array)
    
            # Plot the line first to capture color
            line, = axs[0].plot(range(time_steps), mean_array, label=f"state {state}")
            color = line.get_color()
            self.state_colors.append(color)
    
            # Correct color applied to the shaded region
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

        
    def plot_distributions_over_time_set(self, steps_to_plot=[0, 10, 25, 50]):
        from scipy.stats import norm
        x = np.linspace(self.action_min - 0.5, self.action_max + 0.5, 500)
    
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
    
        plt.tight_layout()
        plt.show()
        
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
            ax.set_ylim(0, y_max)
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
                fills[state].remove()
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

    def all_plots_set(self):
        
        self.plots_set()
        self.plot_distributions_over_time_set()
        #anim = self.animate_distributions_set(interval=50, save_path='RL_animation.gif')

    # ******************************************




