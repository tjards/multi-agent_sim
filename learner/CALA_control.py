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
num_states      = 3             # number of states
action_min      = 0.5 #-1            # minimum of action space
action_max      = 2 # 1             # maximum of action space

# (artificial, just for debugging. would come from env)
target_actions = np.random.uniform(action_min, action_max, num_states) # randomly assigned target actions

#%% Hyperparameters
# -----------------
learning_rate   = 0.1       # rate at which policy updates
variance        = 0.2       # initial variance
variance_ratio  = 1         # default 1, permits faster/slower variance updates
variance_min    = 0.001     # default 0.001, makes sure variance doesn't go too low

# initial means and variances
#means = np.random.uniform(action_min, action_max, num_states)
#variances = np.full(num_states, variance)

#%% Learning Class
# ----------------
class CALA:
    
    # initialize
    #def __init__(self, num_states, action_min, action_max, learning_rate, means, variances):
    def __init__(self, num_states=3):
        
        # load parameters into class
        self.num_states     = num_states
        self.action_min     = action_min
        self.action_max     = action_max
        self.learning_rate  = learning_rate
        self.means          = np.random.uniform(action_min, action_max, num_states) #means
        self.variances      = np.full(num_states, variance) #variances
        
        # store stuff
        self.mean_history       = []
        self.variance_history   = []
        self.reward_history     = []
        
        # store the configs
        configs_tools.update_configs('CALA', [
            ('num_states', num_states),
            ('action_min', action_min),
            ('action_max', action_max),
            ('learning_rate', learning_rate)
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

    # run the simulation
    def run(self, num_episodes, environment):
  
        # note: 'environment' is a function (substitute with actual environment feedback)      
  
        # for the desired number of episodes
        for _ in range(num_episodes):
            
            # initialize local storage 
            mean_store      = []
            variance_store  = []
            reward_store    = []
            
            # for each state
            for state in range(0, self.num_states):
                
                # select the action (based on current mean/variance)
                action = self.select_action(state)
                
                # collect reward (based on feedback from environment)
                reward = environment(state, action)
                
                # update the policy (based on reward and hyperparameters)
                self.update_policy(state, action, reward)
                
                # store 
                mean_store.append(self.means[state])
                variance_store.append(self.variances[state])
                reward_store.append(reward)
            
            # append local storage to history
            self.mean_history.append(mean_store)
            self.variance_history.append(variance_store)
            self.reward_history.append(reward_store)

    # still plots
    # -----------
    def plots(self):
 
        time_steps  = len(self.mean_history)
        fig, axs    = plt.subplots(3, 1, figsize=(10, 12))
        
        # arrayerize the history lists
        mean_array      = np.array(self.mean_history)
        variance_array  = np.array(self.variance_history)
        reward_array    = np.array(self.reward_history)
        
        self.state_colors = []

        # Means
        # ----
        for state in range(self.num_states):
            # plot the means
            line, = axs[0].plot(range(time_steps), mean_array[:, state], label=f"state {state}")
            line_color = line.get_color()
            self.state_colors.append(line_color)  # Store the color
            axs[0].axhline(y=target_actions[state], color = line_color, linestyle='--')
            std_devs = np.sqrt(variance_array[:, state])
            axs[0].fill_between(np.arange(time_steps), mean_array[:, state] - std_devs, mean_array[:, state] + std_devs, color=line_color, alpha=0.3)
        # format the plots    
        axs[0].set_title('Action means over time')
        axs[0].set_xlabel('Episodes')
        axs[0].set_ylabel('Mean with standard deviation')
        axs[0].set_ylim(action_min, action_max)
        axs[0].legend()

        # Variances
        # ---------
        for state in range(self.num_states):
            axs[1].plot(range(time_steps), variance_array[:, state], label=f"state {state}")
        axs[1].set_title('Action variance over time')
        axs[1].set_xlabel('Episodes')
        axs[1].set_ylabel('Variance')
        axs[1].legend()

        # Rewards
        # -------
        for state in range(self.num_states):
            axs[2].plot(range(time_steps), reward_array[:, state], label=f"state {state}")
        axs[2].set_title('Reward over time')
        axs[2].set_xlabel('Episodes')
        axs[2].set_ylabel('Reward')
        axs[2].legend()

        plt.tight_layout()
        plt.show()
      
    # plot Gaussian curves (PDFs) 
    # ---------------------------
    def plot_distributions_over_time(self, steps_to_plot=[0, 100, 250, 500, 750, 999]):
        from scipy.stats import norm
        x = np.linspace(self.action_min - 0.5, self.action_max + 0.5, 500)

        fig, axs = plt.subplots(self.num_states, 1, figsize=(10, 3 * self.num_states))
        #plt.subplots_adjust(hspace=0.4)  # Add this line to increase vertical spacing

        mean_array = np.array(self.mean_history)
        variance_array = np.array(self.variance_history)

        for state in range(self.num_states):
            ax = axs[state] if self.num_states > 1 else axs
            color = self.state_colors[state] if hasattr(self, 'state_colors') else None

            for idx, step in enumerate(steps_to_plot):
                mu = mean_array[step, state]
                var = variance_array[step, state]
                sigma = np.sqrt(var)
                y = norm.pdf(x, mu, sigma)

                # Dynamic alpha for temporal fading
                alpha = 0.1 + 0.8 * (idx / (len(steps_to_plot) - 1))

                # Plot shaded area
                ax.fill_between(x, y, color=color, alpha=alpha, label=f"step {step}" if idx == len(steps_to_plot)-1 else None)

            ax.set_title(f"State {state} - Action distribution evolution")
            ax.set_xlim(self.action_min - 0.5, self.action_max + 0.5)
            ax.set_xlabel("Action value")
            ax.set_ylabel("Probability density")
            #ax.legend(loc='upper right')

        plt.tight_layout()
        plt.show()
        
    # show animation 
    # ------------------    
    def animate_distributions(self, interval=50, save_path=None):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from scipy.stats import norm
        import numpy as np
    
        mean_array = np.array(self.mean_history)
        variance_array = np.array(self.variance_history)
        time_steps = len(mean_array)
    
        x = np.linspace(self.action_min - 0.5, self.action_max + 0.5, 500)
    
        # Compute global max PDF height for consistent y-axis limits
        y_max = 0
        for frame in range(time_steps):
            for state in range(self.num_states):
                mu = mean_array[frame, state]
                sigma = np.sqrt(variance_array[frame, state])
                if sigma > 0:
                    peak = norm.pdf(mu, mu, sigma)
                    y_max = max(y_max, peak)
        y_max *= 1.1  # add 10% headroom
    
        fig, axs = plt.subplots(self.num_states, 1, figsize=(10, 3 * self.num_states))
        plt.subplots_adjust(hspace=0.4)  # Add this line to increase vertical spacing
        
        if self.num_states == 1:
            axs = [axs]
    
        lines = []
        fills = []
    
        for state in range(self.num_states):
            color = self.state_colors[state] if hasattr(self, 'state_colors') else None
            ax = axs[state]
            line, = ax.plot([], [], color=color)
            fill = ax.fill_between(x, np.zeros_like(x), np.zeros_like(x), color=color, alpha=0.3)
            lines.append(line)
            fills.append(fill)
            ax.set_xlim(self.action_min - 0.5, self.action_max + 0.5)
            ax.set_ylim(0, y_max)
            ax.set_title(f"State {state} - Action Probability Density over Time")
            #ax.set_xlabel("Action value")
            #ax.set_ylabel("Probability Density")
    
        # Add time label to the top subplot
        time_text = axs[0].text(0.95, 0.95, '', transform=axs[0].transAxes,
                                ha='right', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
        def init():
            for line in lines:
                line.set_data([], [])
            time_text.set_text('')
            return lines + [time_text]
    
        def update(frame):
            for state in range(self.num_states):
                mu = mean_array[frame, state]
                sigma = np.sqrt(variance_array[frame, state])
                y = norm.pdf(x, mu, sigma)
                lines[state].set_data(x, y)
    
                fills[state].remove()
                fills[state] = axs[state].fill_between(x, y, color=self.state_colors[state], alpha=0.3)
    
            time_text.set_text(f"Time step: {frame}/{time_steps}")
            return lines + fills + [time_text]
    
        ani = animation.FuncAnimation(fig, update, frames=time_steps,
                                      init_func=init, blit=False, interval=interval)
    
        self.ani = ani
    
        if save_path:
            ani.save(save_path, writer='pillow', fps=1000 // interval)
            #ani.save(save_path, writer='ffmpeg', fps=1000 // interval)
        else:
            plt.tight_layout()
            plt.show()
    
        return ani





#%% Example
# --------

def environment(state, action):

    # reward gets exponentially higher, the closer action is to target action
    reward = np.exp(-np.abs(target_actions[state] - action))
    
    return reward

# run the simulation

'''automata = CALA(num_states, action_min, action_max, learning_rate, means, variances)
automata.run(num_episodes=1000, environment=environment)
automata.plots()
#automata.plot_distributions_over_time()
anim = automata.animate_distributions(interval=50, save_path='animation.gif')'''

