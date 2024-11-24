#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 17:29:35 2024

This program implments a method for estimating potential function gradients of neighbouring agents


@author: tjards
"""

'''
# these things come from environment
potential_function: Function \phi(x_ij) representing the potential
gradient_function: Gradient function of the potential, -∇φ(x_ij)

# this will be estimated through data methods
hessian_function: Hessian function of the potential, ∇²φ(x_ij).
'''


# %% import stuff
import numpy as np
import copy

#%% hyperparameters
gain = 0.5




#%%
class GradientEstimator:
    
    # initialize
    def __init__(self, nAgents, dimens, Ts):
        
        """
        Arguments:
            nAgents = number of agents
            Ts      = step size
            dimens  = numbers of dimensions (2D, 3D)
        """
        self.nAgents            = nAgents
        self.Ts                 = Ts
        self.dimens             = dimens
        self.gradient_estimates = np.zeros((nAgents, nAgents, dimens))  # Assume 2D space

    # update gradient estimates based on relative pos/vel
    def update_estimates(self, states_q, states_p, observed_gradients, observed_hessians, A, k_node):
        
        """
        Arguments:
            states_q    = array of agent positions, shape (dimens, nAgents)
            states_p    = array of agent velocities, shape (dimens, nAgents)
            observed_gradients   = gradients observed by this agent: ((dimens, nAgents))
            hessians             = hessians  observed by this agent (this will be from a data-driven model later) ((dimens, nAgents))
            A           = adjacency matrix
            k_node      = index for this agent 

        Returns:
            Updated gradient estimates for all agent pairs.
        """
        # transpose for now, but don't do this in later vers
        #states_q = np.transpose(states_q)
        #states_p = np.transpose(states_p)
        
        # for each neighbour
        for k_neigh in range(0, self.nAgents):
            
            # check if the neighbour is in range
            # ---------------------------------
            
            if k_neigh != k_node:
            
                # use adjacency matrix (new)
                if A[k_node,k_neigh] == 0:
                    in_range = False
                else:
                    in_range = True 
            
                # if within range
                # ---------------
                if in_range:
                    
                    # relative states
                    x_ij = states_q[0:self.dimens,k_neigh] - states_q[0:self.dimens,k_node]
                    v_ij = states_p[0:self.dimens,k_neigh] - states_p[0:self.dimens,k_node]
                
                    # pull current gradient 
                    current_gradient = self.gradient_estimates[k_node, k_neigh]
                    
                    # compute hessian (this will be from a data-driven model later)
                    hessian = observed_hessians[0:self.dimens, 0:self.dimens, k_neigh]
                    
                    # gradient_dot
                    gradient_dot = -np.dot(hessian, v_ij)
                    
                    # predict the next gradient
                    predicted_gradient = current_gradient + gradient_dot * self.Ts
                    
                    # pull the observed gradient 
                    observed_gradient = observed_gradients[0:self.dimens, k_neigh]
                    
                    # correct with innovation
                    updated_gradient = predicted_gradient + gain * (observed_gradient - predicted_gradient)
                    
                    # load
                    self.gradient_estimates[k_node, k_neigh] = updated_gradient
                
            
        
        
        # for i in range(self.num_agents):
        #     for j in range(self.num_agents):
        #         if i != j:
        #             # Compute relative position and velocity
        #             x_ij = positions[j] - positions[i]
        #             v_ij = velocities[j] - velocities[i]

        #             # Current gradient estimate
        #             current_gradient = self.gradient_estimates[i, j]

        #             # Compute the Hessian and update gradient estimate
        #             hessian = self.hessian_function(x_ij)
        #             gradient_dot = -np.dot(hessian, v_ij)

        #             # Predict next gradient
        #             predicted_gradient = current_gradient + gradient_dot * self.dt

        #             # Correct using gradient observation
        #             observed_gradient = self.gradient_function(x_ij)
        #             #gain = 0.5  # Gain for correction, can be tuned
        #             updated_gradient = predicted_gradient + gain * (observed_gradient - predicted_gradient)

        #             # Store updated estimate
        #             self.gradient_estimates[i, j] = updated_gradient

        # return self.gradient_estimates

#%% Example usage with a simple potential function
def potential_function(x):
    """Example Lennard-Jones potential."""
    r = np.linalg.norm(x)
    return 4 * ((1 / r)**12 - (1 / r)**6)

def gradient_function(x):
    """Gradient of the Lennard-Jones potential."""
    r = np.linalg.norm(x)
    if r == 0:
        return np.zeros_like(x)
    factor = 4 * (12 / r**13 - 6 / r**7)
    return factor * x / r

def hessian_function(x):
    """Hessian of the Lennard-Jones potential."""
    r = np.linalg.norm(x)
    if r == 0:
        return np.zeros((dimens, dimens))
    factor1 = 4 * (156 / r**14 - 42 / r**8)
    factor2 = -4 * (12 / r**14 - 6 / r**8)
    outer = np.outer(x, x) / r**2
    return factor1 * outer + factor2 * np.eye(2)


#%% try simulation 
import random
random.seed(42)


nAgents = 10
dimens  = 2
Ts      = 0.02
A = np.ones((nAgents,nAgents)) # assume all connected (would depend on swarm topology)

# initialize states 
states_q = np.random.rand(dimens, nAgents) * 10  # Random positions
states_p = np.random.rand(dimens, nAgents) - 0.5 # Random positions

# initialize the estimator
estimator = GradientEstimator(nAgents, dimens, Ts) 


# # Simulation setup
# num_agents = 3
# positions = np.random.rand(num_agents, 2) * 10  # Random positions
# velocities = np.random.rand(num_agents, 2) - 0.5  # Random velocities
# dt = 0.1

# # Initialize the estimator
# estimator = GradientEstimator(
#     num_agents=num_agents,
#     dt=dt,
#     potential_function=potential_function,
#     gradient_function=gradient_function,
#     hessian_function=hessian_function,
# )

# Run gradient estimation for a few steps
steps = 10
actual_gradients_list = []
estimated_gradients_list = []

for step in range(0, steps):
    
    # for all agents
    for k_node in range(0,nAgents):
        
        # initialie observations 
        observed_gradients = np.zeros((dimens, nAgents))
        observed_hessians = np.zeros((dimens, dimens, nAgents))
        
        # initialize some controls
        controller = 0.1
        cmds = np.zeros((dimens, 1))
        
        # for each neighbour
        for k_neigh in range(0,nAgents):
            # compute relative states
            x_ij = states_q[0:dimens, k_neigh] - states_q[0:dimens, k_node]
            v_ij = states_p[0:dimens, k_neigh] - states_p[0:dimens, k_node]
            observed_gradients[0:dimens, k_neigh]               = gradient_function(x_ij)
            observed_hessians[0:dimens, 0:dimens, k_neigh]      = hessian_function(x_ij)
            
            # compute a command
            cmds += observed_gradients[0:dimens, k_neigh].reshape((dimens,1))
        
        # # simulate the observed gradients 
        # x_ij = positions[j] - positions[i]
        # v_ij = velocities[j] - velocities[i]
        
        # update the estimates 
        estimator.update_estimates(states_q, states_p, observed_gradients, observed_hessians, A, k_node)
    
        # Update positions and velocities (simple dynamics for testing)
        states_q[0:dimens, k_node] += states_p[0:dimens, k_node] * Ts
        states_p[0:dimens, k_node] += controller*cmds.ravel()*Ts
        
    # STORE
    # =====
    actual_gradients = np.zeros_like(estimator.gradient_estimates)
    estimated_gradients = estimator.gradient_estimates
    for k_node in range(nAgents):
        for k_neigh in range(nAgents):
            #if A[k_node, k_neigh] == 1:  # Only compare if they are neighbors
            x_ij = states_q[0:dimens, k_neigh] - states_q[0:dimens, k_node]
            actual_gradients[k_node, k_neigh] = gradient_function(x_ij)
    
    # Store both actual and estimated gradients for later plotting
    actual_gradients_list.append(copy.deepcopy(actual_gradients))
    estimated_gradients_list.append(copy.deepcopy(estimated_gradients))
    



#%% plot stuff
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# convert to arrays
actual_gradients_array = np.array(actual_gradients_list)
estimated_gradients_array = np.array(estimated_gradients_list)

# Create the plot
plt.figure(figsize=(12, 8))
colormap = cm.get_cmap('tab10', nAgents)  # Use 'tab10' for distinct colors, with up to nAgents colors


# Loop through each agent to plot its gradients for all steps
which_agent = 1
for i in range(which_agent):
    
    for j in range(nAgents):
        #if A[i, j] == 1:  # Only plot if agents are neighbors
        
        actual_gradients_for_pair = actual_gradients_array[:, i, j]
        estimated_gradients_for_pair = estimated_gradients_array[:, i, j]
            
        color = colormap(j)  # Get the color for agent j

        #plt.plot(np.arange(0, steps), np.linalg.norm(actual_gradients_for_pair, axis=1), label=f'Actual Gradient (Agent {i+1} -> Agent {j+1})', linestyle='-', color = color)
        #plt.plot(np.arange(0, steps), np.linalg.norm(estimated_gradients_for_pair, axis=1), label=f'Estimated Gradient (Agent {i+1} -> Agent {j+1})', linestyle='--', color = color)
        plt.plot(np.arange(0, steps), np.linalg.norm(actual_gradients_for_pair-estimated_gradients_for_pair, axis=1), label=f'Agent {i+1} -> Agent {j+1}', linestyle='-', color = color)
        


plt.xlabel('Step')
plt.ylabel('Error in Gradient Magnitude')
plt.title('Comparison of Actual vs Estimated Gradients')
#plt.ylim([0, 1])
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#%%% Estimating the Hessian

'''
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt

# Generate synthetic data (e.g., a simple potential function)
def potential_function(x):
    """Simple example: A Lennard-Jones-like potential."""
    return 4 * ((1 / x)**12 - (1 / x)**6)

# Generate sample data (positions and corresponding potential values)
positions = np.linspace(1, 10, 100).reshape(-1, 1)
potential_values = potential_function(positions)

# Define the kernel (RBF kernel with constant term)
kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))

# Instantiate the GaussianProcessRegressor
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Fit the model to the data
gp.fit(positions, potential_values)

# Make predictions for a new range of positions (e.g., 1 to 10)
x_new = np.linspace(1, 10, 1000).reshape(-1, 1)
y_pred, sigma = gp.predict(x_new, return_std=True)

# Plot the predicted potential and uncertainty
plt.figure(figsize=(10, 6))
plt.plot(x_new, y_pred, 'r:', label='Predicted potential')
plt.fill_between(x_new.flatten(), y_pred - sigma, y_pred + sigma, color='gray', alpha=0.2, label='Uncertainty')
plt.scatter(positions, potential_values, c='black', label='Observed data')
plt.title("Gaussian Process Regression for Potential Function")
plt.xlabel("Position")
plt.ylabel("Potential")
plt.legend()
plt.show()

# Compute the gradient (derivative) of the potential using the GPR model
gradient_pred = gp.predict(x_new, return_std=False, evaluate_mse=False)  # derivative w.r.t x

# Plot the gradient
plt.figure(figsize=(10, 6))
plt.plot(x_new, gradient_pred, 'b-', label='Predicted Gradient')
plt.title("Predicted Gradient (1st Derivative) of the Potential Function")
plt.xlabel("Position")
plt.ylabel("Gradient")
plt.legend()
plt.show()

# Compute the Hessian (second derivative of the potential)
# The second derivative can be estimated as the derivative of the gradient
# Since GPR doesn't directly give the second derivative, we use finite difference to approximate
hessian_pred = np.gradient(gradient_pred, x_new.flatten())

# Plot the Hessian (second derivative)
plt.figure(figsize=(10, 6))
plt.plot(x_new, hessian_pred, 'g-', label='Predicted Hessian (2nd Derivative)')
plt.title("Predicted Hessian (2nd Derivative) of the Potential Function")
plt.xlabel("Position")
plt.ylabel("Hessian")
plt.legend()
plt.show()

'''
