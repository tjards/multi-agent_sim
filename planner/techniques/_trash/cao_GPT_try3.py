import numpy as np
import matplotlib.pyplot as plt

# Constants for the potential function and control
R = 10.0
H_bar = 5.0
imath = 1.0

class Agent:
    def __init__(self, position, velocity, neighbors):
        self.position = position  # Position of the agent
        self.velocity = velocity  # Velocity of the agent
        self.neighbors = neighbors  # List of neighboring agent indices

class FlockingControl:
    def __init__(self, agents, malicious_agent_index):
        self.agents = agents  # List of all agents in the swarm
        self.malicious_agent_index = malicious_agent_index  # Index of the malicious agent
        self.vFif = np.zeros(2)  # Filtered velocity
        self.CFif = np.zeros((2, 2))  # Filtered CFif matrix
        self.k_hat = np.zeros(2)  # Estimated parameters

    def update_filter(self, agent_index, a):
        agent = self.agents[agent_index]
        self.vFif = -a * self.vFif + agent.velocity
        self.CFif = -a * self.CFif + np.outer(agent.velocity, agent.velocity)

    def adaptive_update(self, agent_index, Gamma_k):
        agent = self.agents[agent_index]
        v_i = agent.velocity
        C_i = self.CFif
        v_j = np.array([self.agents[j].velocity for j in agent.neighbors])
        v_j_mean = np.mean(v_j, axis=0)

        # Ensure dimensions are compatible
        if Gamma_k.shape[0] != v_i.shape[0]:
            raise ValueError(f"Dimension mismatch: Gamma_k shape {Gamma_k.shape} vs v_i shape {v_i.shape}")

        term1 = Gamma_k @ (v_j_mean - v_i)
        term2 = Gamma_k @ (self.CFif @ self.k_hat + v_i - self.vFif)

        # Ensure k_hat is compatible
        self.k_hat = (self.k_hat - (term1 - term2)).reshape(self.k_hat.shape)

    def potential_function(self, xij, xij_star, R, H_bar, imath):
        delta_ij = np.linalg.norm(xij_star)
        norm_xij = np.linalg.norm(xij)
        return (np.linalg.norm(xij - xij_star)**2 / (R - norm_xij + 1e-6) + 
                (R - delta_ij)**2 / (H_bar + imath) + 
                np.linalg.norm(xij - xij_star)**2 / (norm_xij + delta_ij**2 / (H_bar + imath)))

    def controller_layer2(self, agent_index, kappa_v, kappa_x, C_if, k_hat):
        agent = self.agents[agent_index]
        control_signal = -kappa_v * np.sum([self.agents[n].velocity - agent.velocity for n in agent.neighbors], axis=0)
        control_signal -= kappa_x * np.sum([self.potential_function(self.agents[n].position, agent.position, R, H_bar, imath) for n in agent.neighbors], axis=0)

        # Ensure dimensions are compatible
        if C_if.shape[0] != k_hat.shape[0]:
            raise ValueError(f"Dimension mismatch: C_if shape {C_if.shape} vs k_hat shape {k_hat.shape}")

        control_signal -= C_if @ k_hat
        return control_signal

    def controller_layer3(self, agent_index, alpha_kp, gamma_kp):
        agent = self.agents[agent_index]

        # Ensure alpha_kp is an array with length equal to the number of neighbors
        if len(alpha_kp) != len(agent.neighbors):
            raise ValueError(f"Length of alpha_kp ({len(alpha_kp)}) does not match number of neighbors ({len(agent.neighbors)})")

        control_signal = -np.sum([alpha_kp[n] * np.sign(agent.velocity - self.agents[agent.neighbors[n]].velocity) 
                                  for n in range(len(agent.neighbors))], axis=0)
        return control_signal

    def update_alpha(self, agent_index, gamma_kp):
        agent = self.agents[agent_index]
        alpha_kp = gamma_kp * np.abs(agent.velocity - np.array([a.velocity for a in agent.neighbors]))
        return alpha_kp

def initialize_agents(num_agents, area_size, range_min=2.0, range_max=5.0):
    agents = []
    positions = np.random.rand(num_agents, 2) * area_size
    velocities = (np.random.rand(num_agents, 2) - 0.5) * 2.0

    for i in range(num_agents):
        position = positions[i]
        velocity = velocities[i]
        neighbors = []
        
        # Find neighbors
        for j in range(num_agents):
            if i != j and np.linalg.norm(position - positions[j]) < range_max:
                neighbors.append(j)

        if len(neighbors) < 2:
            neighbors.append((i + 1) % num_agents)  # Ensure each agent has at least 2 neighbors

        agents.append(Agent(position=position, velocity=velocity, neighbors=neighbors))

    return agents

def simulate_flocking(T, dt, agents, flocking_control):
    num_steps = int(T / dt)
    positions = np.zeros((num_steps, len(agents), 2))
    velocities = np.zeros((num_steps, len(agents), 2))
    
    for step in range(num_steps):
        for i in range(len(agents)):
            flocking_control.update_filter(i, a=0.1)
            flocking_control.adaptive_update(i, Gamma_k=np.eye(2))

            control_signal_layer2 = flocking_control.controller_layer2(i, kappa_v=1.0, kappa_x=1.0, C_if=np.eye(2), k_hat=flocking_control.k_hat)
            control_signal_layer3 = flocking_control.controller_layer3(i, alpha_kp=np.ones(len(agents)), gamma_kp=1.0)

            agent = agents[i]
            agent.velocity += control_signal_layer2 + control_signal_layer3
            agent.position += agent.velocity * dt
            
            positions[step, i] = agent.position
            velocities[step, i] = agent.velocity
    
    return positions, velocities

# Set parameters
num_agents = 7
area_size = 100
T = 30
dt = 0.1

agents = initialize_agents(num_agents, area_size)
flocking_control = FlockingControl(agents, malicious_agent_index=0)

positions, velocities = simulate_flocking(T, dt, agents, flocking_control)

# Plot the results
plt.figure(figsize=(10, 8))
for i in range(num_agents):
    plt.plot(positions[:, i, 0], positions[:, i, 1], label=f'Agent {i}')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Agent Trajectories')
plt.legend()
plt.show()




