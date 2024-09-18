import numpy as np

class Agent:
    def __init__(self, position, velocity, neighbors):
        self.position = position  # Position of the agent
        self.velocity = velocity  # Velocity of the agent
        self.neighbors = neighbors  # List of neighboring agents

class FlockingControl:
    def __init__(self, agents, malicious_agent_index):
        self.agents = agents  # List of all agents in the swarm
        self.malicious_agent_index = malicious_agent_index  # Index of the malicious agent
        self.vFif = np.zeros_like(agents[0].velocity)  # Filtered velocity (Eqn (11))
        self.CFif = np.eye(len(agents[0].velocity))  # Filtered Cif (Eqn (12)), initialized as identity matrix
        self.k_hat = np.zeros(len(agents[0].velocity))  # Estimated parameters (Eqn (14))
    
    def update_filter(self, agent_index, a):
        # Update filtered velocity and Cif based on the current agent's state
        # Implements Eqn (11) for velocity and Eqn (12) for Cif
        agent = self.agents[agent_index]
        self.vFif = -a * self.vFif + agent.velocity  # Low-pass filter for velocity
        # For CFif, we need to ensure it's correctly updated based on your actual model
        self.CFif = -a * self.CFif + self.CFif  # Low-pass filter for Cif

    def adaptive_update(self, agent_index, Gamma_k):
        # Update parameter estimates based on filtered values and local observations
        # Implements Eqn (14) for adaptive update of parameter estimates
        agent = self.agents[agent_index]
        v_i = agent.velocity
        C_i = self.CFif  # Ensure C_i matches dimensions for multiplication

        # Get neighbors' velocities
        neighbors = [a for a in self.agents if a != agent]
        if not neighbors:
            # No neighbors; skip update or handle as needed
            return
        
        v_j = np.array([a.velocity for a in neighbors])
        
        # Ensure v_j is correctly shaped
        if v_j.ndim == 1:
            v_j = np.expand_dims(v_j, axis=0)
        
        v_j_mean = np.mean(v_j, axis=0)  # Mean velocity of neighbors

        # Ensure correct dimensions for the matrix operations
        term1 = Gamma_k @ (v_j_mean - v_i)  # Ensure Gamma_k and (v_j_mean - v_i) are compatible
        term2 = Gamma_k @ (self.CFif @ self.k_hat + v_i - self.vFif)  # Ensure self.CFif @ self.k_hat is compatible
        
        # Update self.k_hat
        self.k_hat -= term1 - term2

    def potential_function(self, xij, xij_star, R, H_bar, imath):
        # Calculate the potential function to guide agent behavior
        # Implements the potential function as described in Eqn (16)
        delta_ij = np.linalg.norm(xij_star)  # Distance to the desired configuration
        return (np.linalg.norm(xij - xij_star)**2 / (R - np.linalg.norm(xij)) + 
                (R - delta_ij)**2 / (H_bar + imath) + 
                np.linalg.norm(xij - xij_star)**2 / (np.linalg.norm(xij) + delta_ij**2 / (H_bar + imath)))

    def controller_layer2(self, agent_index, kappa_v, kappa_x, C_if, k_hat):
        # Control signal for agents in Layer 2
        # Implements Eqn (15), which uses information from neighbors in Layer 2 and the malicious agent
        agent = self.agents[agent_index]
        # Compute control signals based on velocity and potential gradient
        control_signal = -kappa_v * np.sum([a.velocity - agent.velocity for a in agent.neighbors], axis=0)
        control_signal -= kappa_x * np.sum([self.potential_function(a.position, agent.position, R, H_bar, imath) for a in agent.neighbors], axis=0)
        control_signal -= C_if @ k_hat  # Adjust control signal based on the parameter estimate
        return control_signal

    def controller_layer3(self, agent_index, alpha_kp, gamma_kp):
        # Control signal for agents in Layer 3
        # Implements Eqn (17), which includes a varying gain and potential gradient
        agent = self.agents[agent_index]
        # Compute control signals based on varying gain and potential gradient
        control_signal = -np.sum([alpha_kp * np.sign(agent.velocity - a.velocity) for a in agent.neighbors], axis=0)
        control_signal -= np.sum([np.gradient(self.potential_function(a.position, agent.position, R, H_bar, imath)) for a in agent.neighbors], axis=0)
        return control_signal

    def update_alpha(self, agent_index, gamma_kp):
        # Update alpha values for agents in Layer 3 based on the velocity difference
        # Implements Eqn (17) for updating the varying gain alpha
        agent = self.agents[agent_index]
        alpha_kp = gamma_kp * np.abs(agent.velocity - np.array([a.velocity for a in agent.neighbors]))
        return alpha_kp

# Example usage
# Define parameters (values need to be set according to your application)
a = 0.1  # Filter gain
Gamma_k = np.eye(2)  # Example 2x2 positive-definite matrix
kappa_v = 1.0  # Control gain for velocity
kappa_x = 1.0  # Control gain for potential gradient
R = 1.0  # Radius parameter for potential function
H_bar = 1.0  # Parameter for potential function
imath = 1.0  # Small constant for potential function
gamma_kp = 1.0  # Positive constant for adaptive update of alpha
alpha_kp = np.array([1.0])  # Varying gain for each neighbor

# Create a list of agents with positions and velocities
agents = [Agent(position=np.array([0, 0]), velocity=np.array([0, 0]), neighbors=[])]
flocking_control = FlockingControl(agents, malicious_agent_index=0)

# Update filters and adaptive parameters
flocking_control.update_filter(agent_index=0, a=a)
flocking_control.adaptive_update(agent_index=0, Gamma_k=Gamma_k)

# Calculate control signals
control_signal_layer2 = flocking_control.controller_layer2(agent_index=0, kappa_v=kappa_v, kappa_x=kappa_x, C_if=np.eye(2), k_hat=np.array([1.0, 1.0]))
control_signal_layer3 = flocking_control.controller_layer3(agent_index=0, alpha_kp=alpha_kp, gamma_kp=gamma_kp)

print("Control Signal Layer 2:", control_signal_layer2)
print("Control Signal Layer 3:", control_signal_layer3)



