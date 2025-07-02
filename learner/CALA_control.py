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
learning_rate   = 0.5     # rate at which policy updates
variance        = 0.2       # initial variance
variance_ratio  = 1      # default 1, permits faster/slower variance updates
variance_min    = 0.001     # default 0.001, makes sure variance doesn't go too low
variance_max    = 10         # highest variance 
epsilon         = 1e-8

counter_max = 100             # when to stop accumualating experience in a trial
reward_mode = 'target'       # 'target' = change orientation of swarm to track target 

        
leader_follower = True # true = define a leader; false = consensus-based
leader = 0

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
        #self.asymmetry  /= np.sum(self.asymmetry)  # Normalize to sum to 1
        
        # initialize actions 
        self.action_set = 0*np.ones((num_states))
        self.reference = 0*np.ones((num_states)) # for when references are needed

        # counter        
        self.counter_max    = counter_max 
        #self.counter        = np.random.uniform(0, self.counter_max, num_states).astype(int) - 500 # all agents start at differnt places
        self.counter        = np.zeros(num_states) -500 # all in synch now, but do asynch (above) later

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

    
    
    def wrap2pi(self, angle):
        
        wrapped_angle = (angle + np.pi) % (2 * np.pi) - np.pi
        
        return wrapped_angle
              
    
    def reward_func_angle(self, psi, psi_s):
        
        # hyperparameters
        k = 3 # tunable 
        clip = False

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
   

        
    # uses global frame
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
                #multi_reward = (np.dot(focal_vec, target_vec) + 1) / 2
                
                multi_reward_xy = 0
                multi_reward_xz = 0
                multi_reward_yz = 0
                
                #x-y plane
                focal_xy = np.arctan2(focal_vec[1], focal_vec[0])
                target_xy = np.arctan2(target_vec[1], target_vec[0])
                
                multi_reward_xy = self.reward_func_angle(focal_xy, target_xy)
                
                #x-z plane
                '''
                focal_xz = np.arctan2(focal_vec[0], focal_vec[2])
                target_xz = np.arctan2(target_vec[0], target_vec[2])
                
                multi_reward_xz = self.reward_func_angle(focal_xz, target_xz)
    
                
                #y-z plane
                focal_yz = np.arctan2(focal_vec[1], focal_vec[2])
                target_yz = np.arctan2(target_vec[1], target_vec[2])
                
                multi_reward_yz = self.reward_func_angle(focal_yz, target_yz)
                '''
    
                
                multi_reward = multi_reward_xy + multi_reward_xz + multi_reward_yz
                #multi_reward =  np.divide(multi_reward_xy +  multi_reward_xz +  multi_reward_yz, 3)          
            
            # if not turret or ... etc
            else:
                
                multi_reward = 0.0
          
        # if not  target tracking        
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
    
        # if doing leader
        if leader_follower:
            
            lag     = leader
            lead    = leader
    
        elif neighbours is not None and len(neighbours) > 1:
            idx = list(neighbours).index(state)
            lag = neighbours[(idx - 1) % len(neighbours)]
            lead = neighbours[(idx + 1) % len(neighbours)]
    
        self.share_statistics(state, [lag, lead], 'actions')
        self.share_statistics(state, [lag, lead], 'rewards')


    # seek consensus between neighbouring rewards (state, list[neighbours])
    def share_statistics(self, state, neighbours, which):
        
        # if doing leader
        if leader_follower:
            if state != leader:
                source = neighbours[0]
                if which == 'actions':
                    self.action_set[state] = self.action_set[source]
                elif which == 'rewards':
                    self.means[state]     = self.means[source]
                    self.variances[state] = self.variances[source]
            return 
                
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
            self.means[state]       = updated_means
            self.variances[state]   = updated_variances
            
        elif which  == 'actions':
            updated_actions = self.action_set[state]
            for neighbour in neighbours:
                #self.action_set[state] = alpha_actions * self.action_set[state] + (1-alpha_actions)*self.action_set[neighbour]
                updated_actions = alpha_actions * updated_actions + (1-alpha_actions)*self.action_set[neighbour]
            self.action_set[state] = updated_actions
            
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
        self.variances[state] = min(self.variances[state], variance_max)

    # ****************************
    # ASYCHRONOUS EXTERNAL UPDATES
    # ****************************


    # NOT WORKING YET!
    def update_reward_increment(self, k_node, state, centroid, focal, target, mode):
        
      
        if self.reward_mode == 'target':
            
            reference = 'local'
            
            if reference == 'local':
            
                # compute the heading vector (centered on centroid)
                v_centroid      = centroid[0:3, 0]
                v_focal         = focal[0:3]
                #v_heading       = v_focal - v_centroid
                v_heading       =  v_centroid - v_focal
                
                # compute the target vector (centered on centroid)
                if target.shape[1] == 0:
                    #v_target = -v_centroid
                    v_target = - v_focal
                else:
                    #v_target = target[0:3, 0] - v_centroid
                    v_target = target[0:3, 0] - v_focal
                
                    
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
        
                # normalize
                v1 /= (np.linalg.norm(v1) + epsilon)
                v2 /= (np.linalg.norm(v2) + epsilon)
                
                #angle           = np.arctan2(v1[1], v1[0])          # focal vector angle
                #angle_desired   = np.arctan2(v2[1], v2[0])          # target vector angle
                
                #reward = self.reward_func_angle(angle, angle_desired)
                reward = (np.dot(v1, v2) + 1) / 2

                reward = np.exp(-1 * reward)


            
                '''
                # get action for this mode
                #action = self.action_set[k_node]
                #frame = self.base_set[k_node] # + self.action_set[k_node]
                #action = (action + np.pi) % (2 * np.pi) - np.pi
                #frame = self.action_set[k_node]
                frame_shift = self.reference[k_node]
                
                # rotation axis
                if mode == 'x':
                    #axis_vec = np.array([1.0, 0.0, 0.0])
                    axis_vec = np.array([0.0, 0.0, 1.0])
                    #axis_vec = np.array([0.0, 1, 0])
                    #plane_indices = [1, 2]
                    plane_indices = [1, 2]
                elif mode == 'z':
                    #axis_vec = np.array([0.0, 0.0, 1.0])
                    #axis_vec = np.array([0, 0.0, 1])
                    axis_vec = np.array([1.0, 0.0, 0.0])
                    #plane_indices = [0, 2]  
                    plane_indices = [0, 1]   

                else:
                    raise ValueError("Invalid mode: must be 'x' or 'z'")
        
                # Build quaternion for rotating into local frame
                #q = quat.e2q(frame * axis_vec)
                q = quat.e2q(frame_shift * axis_vec)
                q_inv = quat.quatjugate(q)
        
        
                # Compute world-frame vectors
                centroid_vec = centroid[0:3, 0]
                focal_vec_world = focal[0:3] - centroid_vec
                
                if target.shape[1] == 0:
                    target_vec_world = -centroid_vec
                else:
                    target_vec_world = target[0:3, 0] - centroid_vec
        
                # Rotate both vectors into agent's local frame
                focal_vec_local = quat.rotate(q_inv, focal_vec_world.reshape(3, 1)).ravel()
                target_vec_local = quat.rotate(q_inv, target_vec_world.reshape(3, 1)).ravel()
        
                # Project into relevant 2D plane
                v1 = focal_vec_local[plane_indices]
                v2 = target_vec_local[plane_indices]
        
                # Normalize
                v1 /= (np.linalg.norm(v1) + epsilon)
                v2 /= (np.linalg.norm(v2) + epsilon)
                
                angle = np.arctan2(v1[1], v1[0])         # focal vector angle
                angle_desired = np.arctan2(v2[1], v2[0]) # target vector angle
                
                reward = self.reward_func_angle(angle, angle_desired)
                '''
        
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

