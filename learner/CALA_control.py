#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 19:45:47 2024

This program implements Continuous Action Learning Automata (CALA) to learn
    an arbitrary set of actions for n-states

@author: tjards

Note: in here, angles are -pi to pi

#notes
 - 22 Sep: the two degrees of freedom are fighting eachother, lets
 look to define the error in local frame or something 
 - also, confirm where the x- and z- axis parameters are being defined and pulled from
 -- probbaly make this more obvious, it's hard to follow the action flow right now
- 05 Oct: pausing this for now, going back to lemni_tools to dev autoencoder method


"""


#%% Import stuff
# ---------------
import numpy as np
import matplotlib.pyplot as plt
#import config.configs_tools as configs_tools
from planner.techniques.utils import quaternions as quat
#config_path=configs_tools.config_path
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
#import copy 

import config.config as cfg

'''
Parameters examples:

    # Simulations parameters

    actions_range = 'angular'                   # 'linear', 'angular' (impacts clipping)
    action_min      = -np.pi/4  # 0             # minimum of action space
    action_max      = np.pi/4   # 2*np.pi       # maximum of action space

    # Hyperparameters

    # learning 
    learning_rate   = 0.5 #0.1      # rate at which policy updates
    variance_init   = 0.4 #0.4      # initial variance
    variance_ratio  = 0.5 #0.1      # default 1, permits faster (>1) /slower (<1)  variance updates
    variance_min    = 0.0001         # default 0.001, makes sure variance doesn't go too low
    variance_max    = 10            # highest variance 
    epsilon         = 1e-6
    counter_max     = 100           # when to stop accumualating experience in a trial
    counter_synch   = True          # True = all agents have same counter; False = all agents random counters (not tested)
    counter_delay   = 500           # number of timesteps to delay learning (gives time for dynamic systems to reach equilibrium)

    # exploration
    explore_dirs            = True  # bias learning in certain direction?
    explore_persistence     = 0.7   # if using explore dirs, tune 0.8–0.95 for smoothness

    # MAS coordination 
    leader_follower         = True  # true = define a leader; false = consensus-based (not working yet)
    leader                  = 0

    # rewards
    reward_mode         = 'target'      # 'target' = change orientation of swarm to track target 
    reward_coupling     = 2             # states per agent, default = 2 (only 2 works right now, look at decoupling later) 
    reward_reference    = 'global'      # 'global' (default),   'local' (not working yet)
    reward_form         = 'sharp'       # 'dot'(default),       'angle' (not working yet), 'sharp' (prototype)                    
    reward_k_theta      = 12.0          # for 'sharp'; steepness for 'sharp' (bigger -> harsher)
    momentum            = False         # if using momentum 
    momentum_beta       = 0.8           # beta param for momentum (0 to 1)
    annealing           = False         # anneal variance down with time?
    annealing_rate      = 0.99          # nominally around 0.99
    kicking             = False         # if kicking (stops reward chasing down)
    kicking_factor      = 1.3           # slighly greater than 1
    sigmoidize          = False         # apply sigmoid in latter stages of learning
'''

show_plots = False 

#%% Learning Class
# ----------------
class CALA:
    
    def __init__(self, config):    
        
        CALA_config = cfg.get_config(config, 'learner.CALA')
        agents_config = cfg.get_config(config, 'agents')

        self.num_agents = agents_config.get('nAgents', None)
        self.reward_coupling = CALA_config.get('reward_coupling', None)
        self.num_states = self.num_agents * self.reward_coupling

        self.action_min     = CALA_config.get('action_min', None)
        self.action_max     = CALA_config.get('action_max', None)
        self.learning_rate  = CALA_config.get('learning_rate', None)
        self.means          = 0.75*np.random.uniform(self.action_min, self.action_max, self.num_states)    #means
        self.variance_init   = CALA_config.get('variance_init', None)
        self.variances      = np.full(self.num_states, CALA_config.get('variance_init', None))  
        self.variance_ratio = CALA_config.get('variance_ratio', None)
        self.variance_min = CALA_config.get('variance_min', None) 
        self.variance_max = CALA_config.get('variance_max', None)                         #variances
        self.prev_update    = np.zeros(self.num_states)              # previous update (used for momentum)
        self.prev_reward    = np.zeros(self.num_states)              # previous reward (used for kicking)
        self.explore_dirs   = np.zeros_like(self.means)        # used for directional exploration
        self.explore_dirs_option = CALA_config.get('explore_dirs', None) # how to set exploration direction (if using)
        self.explore_persistence = CALA_config.get('explore_persistence', None) # how much to persist in exploration direction (if using)

        # inject non-uniform influence or bias across states (used for MAS coordination)
        self.asymmetry  = np.random.rand(self.num_states)
        #self.asymmetry  /= np.sum(self.asymmetry)  # Normalize to sum to 1
        
        # initialize actions
        self.actions_range = CALA_config.get('actions_range', None)
        self.action_set = 0*np.ones((self.num_states))
        self.reference = 0*np.ones((self.num_states)) # for when references are needed

        # counter        
        self.counter_max    = CALA_config.get('counter_max', None)
        self.counter_synch   = CALA_config.get('counter_synch', None)
        self.counter_delay   = CALA_config.get('counter_delay', None)


        if self.counter_synch:    
            self.counter = np.zeros(self.num_states) - self.counter_delay 
        else:
            self.counter = np.random.uniform(0, self.counter_max, self.num_states).astype(int) - self.counter_delay
            
        # store environment variables throughout the trial
        self.reward_mode      = CALA_config.get('reward_mode', None)
        self.reward_reference   = CALA_config.get('reward_reference', None)
        self.reward_form        = CALA_config.get('reward_form', None)
        self.reward_k_theta     = CALA_config.get('reward_k_theta', None)
        self.environment_vars = np.zeros(self.num_states)        

        # other
        self.leader_follower = CALA_config.get('leader_follower', None) 
        self.leader         = CALA_config.get('leader', None)
        self.sigmoidize      = CALA_config.get('sigmoidize', None)

        self.momentum        = CALA_config.get('momentum', None)
        self.momentum_beta   = CALA_config.get('momentum_beta', None)
        self.annealing        = CALA_config.get('annealing', None)
        self.annealing_rate   = CALA_config.get('annealing_rate', None)
        self.kicking          = CALA_config.get('kicking', None)
        self.kicking_factor     = CALA_config.get('kicking_factor', None)

        self.epsilon = CALA_config.get('epsilon', None)

        # store stuff
        self.mean_history       = []
        self.variance_history   = []
        self.reward_history     = []
        self.action_history = []
        

    #%% helper functions
    # ----------------

    # wrap to -pi and pi
    def wrap2pi(self, angle):
        wrapped_angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return wrapped_angle
      
    # angular exponential reward function (experimental - do not use yet)        
    def reward_func_angle(self, psi, psi_s):
        
        # hyperparameters
        k = 1           # tunable 
        clip =False

        # get difference between angles 
        delta = psi - psi_s # actual angle - sight angle (desired)
        
        # wrap -pi to pi
        delta = self.wrap2pi(delta)
        
        # absolute that angle 
        delta = np.abs(delta)
    
        # conditional    
        if delta <= 0.5 * np.pi:
            reward = np.exp(-k * delta)
        else:
            reward = -np.exp(-k * (delta - np.pi))
        
        # clip
        if clip:
            return np.clip(reward, -2, 1)
        else:
            return reward
   
    
    #%% main lemniscate learning
    # --------------------------
    
    def learn_lemni(self, state, state_array, centroid, focal, target, neighbours, mode, allow_ext_reward, ext_reward = 0):
  
        # if permitting external reward
        if allow_ext_reward:
            reward = ext_reward
            
        # define reward internally (default)   
        else:
            reward = self.update_reward_increment(state, state_array, centroid, focal, target, mode)
        
        # update the policy
        self.step(state, reward)
        if self.reward_coupling == 2:
            self.step(state + self.num_agents, reward)
            
        # negotiate with neighbours
        self.negotiate_with_neighbours(state, neighbours)
    
    # inter-agent negotiation
    def negotiate_with_neighbours(self, state, neighbours):
        
        # leader/follower negotiation
        if self.leader_follower:
            
            self.share_statistics(state, [None, None], 'actions')
            self.share_statistics(state, [None, None], 'rewards')
            if self.reward_coupling == 2:
                self.share_statistics(state + self.num_agents, [None, None], 'actions', self.leader + self.num_agents)
                self.share_statistics(state + self.num_agents, [None, None], 'rewards', self.leader + self.num_agents)
    
        # consensus negotation
        elif neighbours is not None and len(neighbours) > 1:
            idx = list(neighbours).index(state)
            lag = neighbours[(idx - 1) % len(neighbours)]
            lead = neighbours[(idx + 1) % len(neighbours)]
            self.share_statistics(state, [lag, lead], 'actions')
            self.share_statistics(state, [lag, lead], 'rewards')
            if self.reward_coupling == 2:
                state += self.num_agents
                lag += self.num_agents
                lead += self.num_agents
                self.share_statistics(state, [lag, lead], 'actions')
                self.share_statistics(state, [lag, lead], 'rewards')
        
    # seek consensus between neighbouring rewards (state, list[neighbours])
    def share_statistics(self, state, neighbours, which, leader_node = 0):
        
        # share leader statistics
        if self.leader_follower:
            if state != leader_node:
                source = leader_node
                if which == 'actions':
                    self.action_set[state] = self.action_set[source]
                elif which == 'rewards':
                    self.means[state]     = self.means[source]
                    self.variances[state] = self.variances[source]
            return 
                
        # share through consensus 
        alpha_rewards  = self.asymmetry[state]                             
        alpha_actions  = self.asymmetry[state] 
        
        if which == 'rewards':
            updated_means       = self.means[state]
            updated_variances   = self.variances[state]
            for neighbour in neighbours:
                #self.means[state]       = alpha_rewards * self.means[state] + (1-alpha_rewards)*self.means[neighbour]
                #self.variances[state]   = alpha_rewards * self.variances[state] + (1-alpha_rewards)*self.variances[neighbour]
                updated_means       = alpha_rewards * updated_means     + (1-alpha_rewards)*self.means[neighbour]
                updated_variances   = alpha_rewards * updated_variances + (1-alpha_rewards)*self.variances[neighbour]
            #self.means[state]       = updated_means
            self.means[state]       = self.wrap2pi(updated_means)
            self.variances[state]   = updated_variances
            
        elif which  == 'actions':
            updated_actions = self.action_set[state]
            for neighbour in neighbours:
                #self.action_set[state] = alpha_actions * self.action_set[state] + (1-alpha_actions)*self.action_set[neighbour]
                updated_actions = alpha_actions * updated_actions + (1-alpha_actions)*self.action_set[neighbour]
            #self.action_set[state] = updated_actions
            self.action_set[state] = self.wrap2pi(updated_actions)

            
    # select action
    def select_action(self, state):
        
        # pull mean and variance for given state
        mean        = self.means[state]
        variance    = self.variances[state]
        
        # if biasing in a direction (smoother)
        if self.explore_dirs_option:
            
            #if self.explore_dirs[state] == 0:
            #    self.explore_dirs[state] = np.random.normal(0, 1)
            
            self.explore_dirs[state] = (
                self.explore_persistence * self.explore_dirs[state] + 
                (1 - self.explore_persistence) * np.random.normal(0, 1))
            
            action = self.means[state] + self.explore_dirs[state] * np.sqrt(self.variances[state])

        else:            
            
            # select action from normal distribution
            action = np.random.normal(mean, np.sqrt(variance))
        
        # return the action, onstrained using clip()
        if self.actions_range == 'linear':
        
            return np.clip(action, self.action_min, self.action_max)
        
        elif self.actions_range == 'angular':
            
            #return np.mod(action, 2 * np.pi)
            #return (action + np.pi) % (2 * np.pi) - np.pi
            action = self.wrap2pi(action)
            return np.clip(action, self.action_min, self.action_max)
           
    
    # update policy 
    def update_policy(self, state, action, reward):
        
        # reward
        # ------
                    
        if self.sigmoidize:     
            
            r_linear = reward

            # compute sigmoid reward
            k = 10          # steepness parameter
            r_sigmoid = 1 / (1 + np.exp(-k * (r_linear - 0.5)))

            # compute blend weight based on variance
            avg_var = np.mean(self.variances)               # average variance across all states
            w = 1 - np.clip(avg_var / self.variance_init, 0, 1)    # weight increases as variance drops

            # Hybrid reward
            reward = (1 - w) * r_linear + w * r_sigmoid


        # distribution
        # ------------
        
        if self.kicking:
            if reward < self.prev_reward[state]-0.05:
                self.variances[state] *= self.kicking_factor
            self.prev_reward[state] = reward
        
        if self.annealing:
            self.variances[state] *= self.annealing_rate

        # pull mean and variance for given state
        mean        = self.means[state]
        variance    = self.variances[state]
        
        # update mean and variance based on reward signal
        if self.momentum:
            delta                   = reward * (action - mean)
            delta                   = self.momentum_beta * self.prev_update[state] + (1 - self.momentum_beta) * delta
            self.prev_update[state] = delta
            self.means[state]       += self.learning_rate * delta
        else:
            self.means[state]       += self.learning_rate * reward * (action - mean)
        
        # after updating self.means[state]
        if self.actions_range == 'angular':
            self.means[state] = self.wrap2pi(self.means[state])
            self.means[state] = np.clip(self.means[state], self.action_min, self.action_max)
        elif self.actions_range == 'linear':
            self.means[state] = np.clip(self.means[state], self.action_min, self.action_max)
                
        self.variances[state]   += self.variance_ratio * self.learning_rate * reward * ((action - mean) ** 2 - variance)
        
        # constaints
        # ----------
        
        self.variances[state] = max(self.variance_min, self.variances[state])
        self.variances[state] = min(self.variances[state], self.variance_max)
        

    # ****************************
    # ASYCHRONOUS EXTERNAL UPDATES
    # ****************************

    def update_reward_increment(self, k_node, state, centroid, focal, target, mode):
        
        if self.reward_mode == 'target':        
            
            # compute the heading vector (centered on centroid)
            v_centroid      = centroid[0:3, 0]
            v_focal         = focal[0:3]
            v_heading       =  v_centroid - v_focal
            
            # compute the target vector (centered on centroid)
            if target.shape[1] == 0:
                v_target = - v_focal
            else:
                v_target = target[0:3, 0] - v_focal
            
            # =============
            # when coupled
            # =============
            
            if self.reward_coupling == 2:
                
                if self.reward_reference == 'global':
                    
                    v1 = v_heading
                    v2 = v_target
            
                    if self.reward_form == 'dot':
        
                        v1 /= (np.linalg.norm(v1) + self.epsilon)
                        v2 /= (np.linalg.norm(v2) + self.epsilon)
                        reward = (np.dot(v1, v2) + 1) / 2
    
    
                    elif self.reward_form == 'angle':
                        
                        reward_sigma = 0.5
                        v1 /= (np.linalg.norm(v1) + self.epsilon)
                        v2 /= (np.linalg.norm(v2) + self.epsilon)
                        angle_diff = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
                        reward = np.exp(-angle_diff**2 / reward_sigma**2)  # Gaussian bump at 0
                        
                    elif self.reward_form == 'sharp': 
                        
                        # helper functions
                        _norm = lambda v: v / (np.linalg.norm(v) + self.epsilon)
                        _angle = lambda a, b: np.arccos(np.clip(np.dot(_norm(a), _norm(b)), -1.0, 1.0))
                        
                        # high-contrast alignment: exp(-k * theta^2)
                        theta = _angle(v1, v2)
                        reward = np.exp(-self.reward_k_theta * theta**2) 
                
                elif self.reward_reference == 'local':
                        
                        print('not done yet')
                       
            # ===============
            # when decoupled
            # ===============
            
            # if this is just one axis (i.e., not representing coupled axes)
            elif self.reward_coupling == 1:
            
                print('not done yet')
                
                # need to do angle wrapping up here 
                '''if reward_reference == 'local':
                    
                    # define world axis 
                    unit_lem = np.array([1, 0, 0]).reshape((3, 1))  # x-dir
                    twist_perp = np.array([0, 0, 1]).reshape((3,1)) # z-dir
                    
                    if mode == 'x':
                    
                        qx = quat.e2q(self.action_set[k_node] * unit_lem.ravel())
                        qz = quat.e2q(self.reference[k_node] * twist_perp.ravel())
                        plane_indices = [1, 2] # not sure about this yet
                        
                    if mode == 'z':
                        
                        qx = quat.e2q(self.reference[k_node] * unit_lem.ravel())
                        qz = quat.e2q(self.action_set[k_node] * twist_perp.ravel())
                        plane_indices = [0, 1] # not sure about this yet
                
                    # compute the total rotation
                    q_total = quat.quat_mult(qz, qx)
                    q_total_ = quat.quatjugate(q_total) # inverse
                    
                    # rotate both on swarm's local frame
                    v_heading_local = quat.rotate(q_total_, v_heading.reshape(3, 1)).ravel()
                    v_target_local  = quat.rotate(q_total_, v_target.reshape(3, 1)).ravel()
                    
                    # project into relevant 2D plane
                    v1 = v_heading_local[plane_indices]
                    v2 = v_target_local[plane_indices]
            
    
                elif reward_reference == 'global':
                    
                    if mode == 'x':
                        v1 = v_heading[[1, 2]]  # project onto y-z plane
                        v2 = v_target[[1, 2]]
                    elif mode == 'z':
                        v1 = v_heading[[0, 1]]  # project onto x-y plane
                        v2 = v_target[[0, 1]]
                
                    
                if reward_form == 'dot':
                    # normalize
                    v1 /= (np.linalg.norm(v1) + epsilon)
                    v2 /= (np.linalg.norm(v2) + epsilon)
                    reward = (np.dot(v1, v2) + 1) / 2
                    
                elif reward_form == 'angle':
                    
                    angle           = np.arctan2(v1[1], v1[0])          # focal vector angle
                    angle_desired   = np.arctan2(v2[1], v2[0])          # target vector angle
                    reward = self.reward_func_angle(angle, angle_desired)'''
    
            return reward
        

    # step counter forward 
    def step(self, state, reward):
  
        # increment counter
        self.counter[state] += 1

        if self.counter[state] >= self.counter_max:
            self.update_policy(state, self.action_set[state], reward)
            self.counter[state] = 0
            self._log_state(state, self.action_set[state], reward)  # log old action
            self.action_set[state] = self.select_action(state)
        else:
            self._log_state(state, self.action_set[state], reward)


    # store current step info into history buffers
    def _log_state(self, state, action, reward):
    
        # expand storage if necessary
        while len(self.mean_history) <= state:
            self.mean_history.append([])
            self.variance_history.append([])
            self.reward_history.append([])
            self.action_history.append([])

    

        # store data for this state
        self.mean_history[state].append(self.means[state])
        self.variance_history[state].append(self.variances[state])
        self.reward_history[state].append(reward)
        self.action_history[state].append(action)

    
    # ************** #
    #   PLOTS        #
    # ************** #
    
   
    def plots_set(self, just_leader = True):
        
        # make sure there is a color map
        #self.state_color_map = {}
        if not hasattr(self, "state_color_map"):
            self.state_color_map = {}
        
        # exit if there is no hostory to plot    
        if not self.mean_history or not self.mean_history[0]:
            print("plots_set(): No history to plot yet.")
            return
        
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        time_steps = len(self.mean_history[0])
        #self.state_colors = []  # reset in case this is run fresh
        #self.state_color_map = {} 
    
        # Choose states to plot
        if just_leader:
            states_to_plot = [self.leader, self.leader + self.num_agents]
        else:
            states_to_plot = list(range(self.num_states))
    
        for state in states_to_plot:
            mean_array = np.array(self.mean_history[state])
            variance_array = np.array(self.variance_history[state])
            reward_array = np.array(self.reward_history[state])
            std_devs = np.sqrt(variance_array)
    
            # plot the line first to capture color
            if just_leader:
                line, = axs[0].plot(range(time_steps), mean_array)
            else:
                line, = axs[0].plot(range(time_steps), mean_array, label=f"state {state}")
            color = line.get_color()
            #self.state_colors.append(color)
            self.state_color_map[state] = color  # <- keyed by actual state index
    
            # correct color applied to the shaded region
            axs[0].fill_between(range(time_steps),
                                mean_array - std_devs,
                                mean_array + std_devs,
                                color=color,
                                alpha=0.3)
    
            if just_leader:
                
                axs[1].plot(range(time_steps), variance_array, color=color)
                axs[2].plot(range(time_steps), reward_array, color='green')
                
            else:    
                axs[1].plot(range(time_steps), variance_array, label=f"state {state}", color=color)
                axs[2].plot(range(time_steps), reward_array, label=f"state {state}", color=color)
    
        axs[0].set_title('Mean with Std Dev')
        axs[1].set_title('Variance')
        axs[2].set_title('Reward')
    
        for ax in axs:
            ax.set_xlabel("Steps")
            if not just_leader:
                ax.legend()
    
        #plt.tight_layout()
        #plt.show()
        plt.tight_layout()
        fig.savefig('visualization/plots/CALA_plots_set.png', dpi=300, bbox_inches='tight')
        if not show_plots:
            plt.close(fig)

   
    def animate_distributions_set(self, interval=50, save_path='visualization/animations/CALA_distributions.gif', just_leader = True):
        
        import matplotlib.animation as animation
        from scipy.stats import norm
        
        # exit of there is no history
        if not self.mean_history or not self.mean_history[0]:
            print("animate_distributions_set(): No history to animate yet.")
            return
        
        # ensure map exists 
        if not hasattr(self, "state_color_map"):
            self.state_color_map = {}
        
    
        time_steps = len(self.mean_history[0])
        x = np.linspace(self.action_min - 0.5, self.action_max + 0.5, 500)
    
        # Choose which states to animate
        if just_leader:
            states_to_plot = [self.leader, self.leader+self.num_agents]
        else:
            states_to_plot = list(range(self.num_states))
    
        # Find global max y for PDF
        y_max = 0
        for state in states_to_plot:
            for t in range(time_steps):
                sigma = np.sqrt(self.variance_history[state][t])
                mu = self.mean_history[state][t]
                y_max = max(y_max, norm.pdf(mu, mu, sigma))
        y_max *= 1.1
    
        fig, axs = plt.subplots(len(states_to_plot), 1, figsize=(10, 3 * len(states_to_plot)))
        plt.subplots_adjust(hspace=0.4)
        if len(states_to_plot) == 1:
            axs = [axs]
    
        lines, fills = [], []
        for idx, state in enumerate(states_to_plot):
            #color = self.state_colors[state] if hasattr(self, 'state_colors') else None
            color = self.state_color_map.get(state, None) if hasattr(self, 'state_color_map') else None

            ax = axs[idx]
            line, = ax.plot([], [], color=color)
            fill = ax.fill_between(x, np.zeros_like(x), np.zeros_like(x), color=color, alpha=0.3)
            lines.append(line)
            fills.append(fill)
            ax.set_xlim(self.action_min - 0.5, self.action_max + 0.5)
            ax.set_ylim(0, y_max)
            #ax.set_ylim(0, 1)
            
            if just_leader:
                
                ax.set_title(f"Action PDF over Time (leader)")
                
            else:
            
                ax.set_title(f"State {state} - Action PDF over Time")
    
        time_text = axs[0].text(0.95, 0.95, '', transform=axs[0].transAxes,
                                ha='right', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
        def init():
            for line in lines:
                line.set_data([], [])
            time_text.set_text('')
            return lines + [time_text]
    
        def update(frame):
            for idx, state in enumerate(states_to_plot):
                mu = self.mean_history[state][frame]
                sigma = np.sqrt(self.variance_history[state][frame])
                y = norm.pdf(x, mu, sigma)
                lines[idx].set_data(x, y)
    
                # remove previous collection from axis
                for artist in axs[idx].collections:
                    artist.remove()
    
                # add new one
                #fills[idx] = axs[idx].fill_between(x, y, color=self.state_colors[state], alpha=0.3)
                fills[idx] = axs[idx].fill_between(x, y, color=self.state_color_map.get(state, None), alpha=0.3)

    
            time_text.set_text(f"Time step: {frame}/{time_steps}")
            return lines + fills + [time_text]
    
        frame_skip = 100
        frames=range(0, time_steps, frame_skip)
    
        ani = animation.FuncAnimation(fig, update, frames,
                                      init_func=init, blit=False, interval=interval)
    
        if save_path:
            ani.save(save_path, writer='pillow', fps=1000 // interval)
            plt.close(fig)
        else:
            plt.tight_layout()
            plt.show()
    
        return ani
    
    
    def plot_exploration_contours(self, just_leader=True, bins=100):
        """
        Plot a 2D contour of explored action space colored by reward.
        """
        # Select which states to plot (leader pair if coupled)
        if just_leader:
            x_actions = self.action_history[self.leader]
            y_actions = self.action_history[self.leader + self.num_agents]
            rewards = self.reward_history[self.leader]  # assume same reward for coupled pair
        else:
            # Combine all states
            x_actions = []
            y_actions = []
            rewards = []
            for i in range(self.num_agents):
                x_actions.extend(self.action_history[i])
                y_actions.extend(self.action_history[i + self.num_agents])
                rewards.extend(self.reward_history[i])  # or avg of pair
    
        x_actions = np.array(x_actions)
        y_actions = np.array(y_actions)
        rewards = np.array(rewards)
    
        # Create 2D histogram weighted by reward
        heatmap, xedges, yedges = np.histogram2d(
            x_actions, y_actions, bins=bins,
            range=[[self.action_min, self.action_max],
                   [self.action_min, self.action_max]],
            weights=rewards
        )
    
        # Normalize
        counts, _, _ = np.histogram2d(
            x_actions, y_actions, bins=bins,
            range=[[self.action_min, self.action_max],
                   [self.action_min, self.action_max]]
        )
        avg_reward = np.divide(heatmap, counts, out=np.zeros_like(heatmap), where=counts>0)
    
        # Plot contour
        X, Y = np.meshgrid(
            np.linspace(self.action_min, self.action_max, bins),
            np.linspace(self.action_min, self.action_max, bins)
        )
    
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Y, avg_reward, levels=50, cmap='viridis')
        plt.colorbar(contour, label="Average Reward")
        plt.scatter(x_actions, y_actions, c=rewards, cmap='coolwarm', s=10, alpha=0.5)
        plt.xlabel("Action X")
        plt.ylabel("Action Y")
        plt.title("Exploration Contours: Actions vs Reward")
        #plt.tight_layout()
        #plt.show()
        plt.tight_layout()
        plt.savefig('visualization/plots/CALA_exploration_contours.png', dpi=300, bbox_inches='tight')
        if not show_plots:
            plt.close()
        
    def plot_exploration_surface(self, just_leader=True, bins=50):
        """
        3D surface plot of reward over explored action space with border outline.
        """
    
        # Select states (X=leader, Y=leader+num_agents if coupled)
        if just_leader:
            x_actions = np.array(self.action_history[self.leader])
            y_actions = np.array(self.action_history[self.leader + self.num_agents])
            rewards = np.array(self.reward_history[self.leader])
        else:
            # Combine all agents
            x_actions, y_actions, rewards = [], [], []
            for i in range(self.num_agents):
                x_actions.extend(self.action_history[i])
                y_actions.extend(self.action_history[i + self.num_agents])
                rewards.extend(self.reward_history[i])
            x_actions, y_actions, rewards = np.array(x_actions), np.array(y_actions), np.array(rewards)
    
        # Create 2D histogram of average rewards
        heatmap, xedges, yedges = np.histogram2d(
            x_actions, y_actions, bins=bins,
            range=[[self.action_min, self.action_max],
                   [self.action_min, self.action_max]],
            weights=rewards
        )
        counts, _, _ = np.histogram2d(
            x_actions, y_actions, bins=bins,
            range=[[self.action_min, self.action_max],
                   [self.action_min, self.action_max]]
        )
        avg_reward = np.divide(heatmap, counts, out=np.zeros_like(heatmap), where=counts > 0)
    
        # Prepare mesh for surface
        X, Y = np.meshgrid(
            (xedges[:-1] + xedges[1:]) / 2,
            (yedges[:-1] + yedges[1:]) / 2
        )
        
        #Z = avg_reward
        Z = np.exp((avg_reward - 1))  # emphasize peaks
        
        # nonzero_values = avg_reward[avg_reward > 0]
        # if nonzero_values.size == 0:
        #     Z = np.zeros_like(avg_reward)
        # else:
        #     min_r = np.nanmin(nonzero_values)
        #     max_r = np.nanmax(nonzero_values)
        #     Z = (avg_reward - min_r) / (max_r - min_r + 1e-6)
        #     Z = Z ** 0.5   # nonlinear boost (adjust exponent)
        #     Z *= 2.0       # vertical exaggeration

    
        # Plot 3D surface
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z.T, cmap='viridis', edgecolor='none', alpha=0.8)
    
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Average Reward')
    
        # Compute and plot convex hull outline (border of explored region)
        points = np.column_stack((x_actions, y_actions))
        if len(points) >= 3:  # Hull requires >= 3 points
            hull = ConvexHull(points)
            border = points[hull.vertices]
            ax.plot(border[:, 0], border[:, 1], np.max(avg_reward) * 1.05, 'r-', lw=2)
    
        # Labels
        ax.set_xlabel("Action X")
        ax.set_ylabel("Action Y")
        ax.set_zlabel("Reward")
        ax.set_title("Exploration Surface with Border")
    
        plt.tight_layout()
        plt.show()

    
    from scipy.spatial import ConvexHull

    
    def plot_exploration_contours_hull(self, just_leader=True, bins=100):
        """
        Plot a 2D contour of explored action space colored by reward,
        excluding unvisited bins and wrapping with a convex hull.
        """
        import matplotlib.cm as cm
        
        #color_map_to_use = 'coolwarm'
        color_map_to_use = 'YlOrRd'
        
        
        # Select which states to plot (leader pair if coupled)
        if just_leader:
            x_actions = self.action_history[self.leader]
            y_actions = self.action_history[self.leader + self.num_agents]
            rewards = self.reward_history[self.leader]  # assume same reward for coupled pair
        else:
            # Combine all states
            x_actions, y_actions, rewards = [], [], []
            for i in range(self.num_agents):
                x_actions.extend(self.action_history[i])
                y_actions.extend(self.action_history[i + self.num_agents])
                rewards.extend(self.reward_history[i])  # or avg of pair
    
        x_actions = np.array(x_actions)
        y_actions = np.array(y_actions)
        rewards = np.array(rewards)
    
        # Create 2D histogram weighted by reward
        heatmap, xedges, yedges = np.histogram2d(
            x_actions, y_actions, bins=bins,
            range=[[self.action_min, self.action_max],
                   [self.action_min, self.action_max]],
            weights=rewards
        )
    
        # Normalize by counts to get average reward per bin
        counts, _, _ = np.histogram2d(
            x_actions, y_actions, bins=bins,
            range=[[self.action_min, self.action_max],
                   [self.action_min, self.action_max]]
        )
        avg_reward = np.divide(heatmap, counts, out=np.zeros_like(heatmap), where=counts > 0)
    
        # Mask zero (unvisited) values for plotting
        avg_reward_masked = np.ma.masked_equal(avg_reward, 0)
    
        # Create grid
        X, Y = np.meshgrid(
            np.linspace(self.action_min, self.action_max, bins),
            np.linspace(self.action_min, self.action_max, bins)
        )
    
        # Plot contour only for visited bins
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Y, avg_reward_masked, levels=50, cmap=color_map_to_use)
        plt.colorbar(contour, label="Average Reward")
    
        # Plot convex hull around explored points
        explored_points = np.column_stack((x_actions, y_actions))
        if len(explored_points) >= 3:
            hull = ConvexHull(explored_points)
            hull_points = explored_points[hull.vertices]
            plt.plot(hull_points[:, 0], hull_points[:, 1], 'g-', linewidth=1)
    
        # Optionally scatter points
        #plt.scatter(x_actions, y_actions, c=rewards, cmap='coolwarm', s=10, alpha=0.4)
        plt.scatter(x_actions, y_actions, c=rewards,
            cmap=color_map_to_use, s=60, alpha=0.1, edgecolors='none')
    
        plt.xlabel("Action X")
        plt.ylabel("Action Y")
        plt.title("Exploration Contours with Convex Hull")
        
        # Find highest reward (ignoring zero/unvisited bins)
        nonzero_rewards = rewards[rewards > 0]
        if nonzero_rewards.size > 0:
            max_idx = np.argmax(nonzero_rewards)
            max_x = x_actions[rewards > 0][max_idx]
            max_y = y_actions[rewards > 0][max_idx]
            max_val = nonzero_rewards[max_idx]
            

            #coolwarm_red = cm.get_cmap(color_map_to_use)(1.0)
            coolwarm_red = cm.get_cmap(color_map_to_use)(max_val)
        
            # Plot marker and label
            plt.plot(max_x, max_y, 'o', color=coolwarm_red, markersize=8, markeredgecolor='green', markeredgewidth=1.5)
            #plt.text(max_x, max_y, f"{max_val:.2f}", color='black', fontsize=10,
            #         ha='left', va='bottom', fontweight='bold')
        
        
        plt.tight_layout()
        plt.savefig('visualization/plots/CALA_exploration_contours_hull.png', dpi=300, bbox_inches='tight')
        plt.close()


    # plots
    def all_plots_set(self, just_leader = True):
        
        just_leader = True
        save_path='visualization/animations/CALA_animation.gif'
        #save_path= None
        
        self.plots_set(just_leader = just_leader)
        #self.plot_distributions_over_time_set()
        self.animate_distributions_set(save_path=save_path, just_leader = just_leader)
        #anim = self.animate_distributions_set(interval=50, save_path='RL_animation.gif')


