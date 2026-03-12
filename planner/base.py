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
    
    def __init__(self, config, **kwargs):

        '''
        Initialize base planner
        subclasses MUST call super().__init__(config, **kwargs). Example:

            class CustomPlanner(BasePlanner):
                def __init__(self, config, **kwargs):
                    super().__init__(config, **kwargs)  # initializes base planner attributes
                    self.custom_parameter = kwargs.get('custom_parameter')

        '''

        # number of agents
        self.config = config 

        # interactions
        self.sensor_range_matrix    = None # matrix representing range at which neighbours can sense: currently, "r_matrix"
        self.interaction_graph      = None # Graph representation for the purpose of interactions: currently, "A"

        # connections 
        self.connection_range_matrix    = None # matrix representing range at which neighbours are connected (could be same as sensor range, but not necessarily): currently, "lattice"
        self.connection_graph           = None # Graph representation for the purpose of connections: currently, "A_connectivity"
        
        # pins
        self.pin_assignments            = None # Agents named as pins: currently, "pin_matrix"

    # ================= #
    # MANDATORY METHODS #
    # ================= #

    @abstractmethod
    def compute_cmd(self, states, targets, index, **kwargs):
        """
        Compute control command for an agent
        Args:
            states: Agents position/velocity 
            targets: Targets positions/velocities (per agent)
            index: Agent index to compute command for 
            **kwargs: Strategy-specific parameters (e.g., obstacle, centroid, trajectory, walls, etc.)
        """
        pass

    # ================ #
    # OPTIONAL METHODS #
    # ================ #

    # we can differentiate between updating the trajectory (states) and producing commands (accelerations)
    def update_trajectory(self, Trajectory, targets, **kwargs):

        '''
         Currently in trajectory.py, but, given dependencies, it might make more sense to have it here
        '''

        # defaults to just copy targets, but can be overridden by other planners
        Trajectory.trajectory = targets.copy() 

        #example: encirclement/lemni also adjusts trajectory.lemni and trajectory.sorted_neighs


    # update the interaction and connection graphs (if required)
    def update_graphs(self, A_interaction, A_connectivity, **kwargs):

        self.interaction_graph = A_interaction
        self.connection_graph = A_connectivity

    # update the pin assignments (if required)
    def update_pins(self, pin_matrix, **kwargs):

        self.pin_assignments = pin_matrix

    # update learning (if required)
    def update_learning(self, **kwargs):

        pass

    # return planner-specific parameters (if required)
    def get_params(self):

        return None





   


