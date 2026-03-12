# base class for planners

'''
Common attributes:

    states and targets: all agents, in format: 
    [   x0, ..., xn-1,
        y0, ..., yn-1,
        z0, ..., zn-1,
        vx0, ..., vxn-1,
        vy0, ..., vyn-1,
        vz0, ..., vzn-1     
    ] 

    obstacles:
    [
        x0, ..., xn-1,
        y0, ..., yn-1,
        z0, ..., zn-1,    
        r0, ..., rn-1    radius   
    ]
'''

# I want to enforce certain things 
from abc import ABC, abstractmethod

class BasePlanner(ABC):
    
    def __init__(self, config, states, *args, **kwargs):

        # number of agents
        self.config = config 
        self.nAgents = states.shape[1]

        # interactions
        self.sensor_range_matrix    = None # matrix representing range at which neighbours can sense: currently, "r_matrix"
        self.interaction_graph      = None # Graph representation for the purpose of interactions: currently, "A"

        # connections 
        self.connection_range_matrix    = None # matrix representing range at which neighbours are connected (could be same as sensor range, but not necessarily): currently, "lattice"
        self.connection_graph           = None # Graph representation for the purpose of connections: currently, "A_connectivity"
        
        # pins
        self.pin_assignments            = None # Agents named as pins: currently, "pin_matrix"

    @abstractmethod
    def compute_cmd(self, states, targets, obstacles, index, *args, **kwargs):

        # index is the agent being examined (typically against interaction graph)

        pass

    def update_graphs(self, A_interaction, A_connectivity, *args, **kwargs):

        self.interaction_graph = A_interaction
        self.connection_graph = A_connectivity

    def update_pins(self, pin_matrix, *args, **kwargs):

        self.pin_assignments = pin_matrix

    def update_learning(self, learned_params, *args, **kwargs):

        pass






   


