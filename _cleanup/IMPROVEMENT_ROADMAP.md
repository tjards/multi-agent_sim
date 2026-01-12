# Multi-Agent Simulator - Improvement Roadmap

**Date:** January 5, 2026  
**Status:** Based on complete architecture analysis  
**Scope:** All modules, learning systems, and data pipeline

---

## Executive Summary

Your simulator has **excellent core functionality** with 8 planning techniques, 4 learning approaches, and sophisticated agent dynamics. However, it suffers from:

1. **Scattered configuration** - Parameters in 10+ places, runtime modifications
2. **Tight learning coupling** - CALA baked into lemni_tools, QL baked into pinning_RL_tools
3. **Inconsistent technique interfaces** - Each technique has different function signatures
4. **Monolithic learner** - CALA_control.py is 1028 lines with mixed concerns
5. **Memory inefficiency** - Entire state history kept in RAM (History class)
6. **No test framework** - Manual validation only
7. **If/elif technique selection** - Orchestrator has conditional imports for each technique

**Good news:** The architecture is sound. With systematic refactoring, this becomes a professional research tool.

---

## Phase 1: QUICK WINS (1-2 weeks)

### 1.1 Setup Project Infrastructure

**Files to create:**
- `requirements.txt` - Pin all dependencies
- `setup.py` / `pyproject.toml` - Modern Python packaging
- `.gitignore` - Exclude `__pycache__`, `*.pyc`, `data/`, venv
- `Makefile` or `justfile` - Common commands

**Effort:** 2-3 hours  
**Impact:** ⭐⭐⭐ (Reproducibility, onboarding)

**Steps:**
1. Identify all imports across codebase
2. Run `pip freeze` to get current versions
3. Create `requirements.txt` with pinned versions
4. Test in fresh venv to validate
5. Add dev dependencies (pytest, black, mypy, etc.)

---

### 1.2 Document Current Configuration

**Files to create:**
- `CONFIG_REFERENCE.md` - All parameters and their sources
- `config/schema.json` - Parameter schema
- `config/examples/` - Example configs for each technique

**Effort:** 2-3 hours  
**Impact:** ⭐⭐⭐ (Understanding, reproducibility)

**Steps:**
1. Grep all files for hardcoded parameters
2. Document parameter sources (main.py, agents.py, technique files, learner files)
3. Create schema showing all parameters
4. Extract working configs to `config/examples/`

---

### 1.3 Clean Up Dead Code

**Files to delete/archive:**
- `planner/techniques/utils/pretrash/` - Old graph tools
- `utils/pretrash/` - Old conic tools
- Any commented-out code blocks >10 lines

**Effort:** 1 hour  
**Impact:** ⭐⭐ (Code clarity)

**Steps:**
1. Review pretrash directories
2. Verify nothing uses them (grep imports)
3. Move to `archive/` branch or commit as "cleanup"
4. Remove commented code

---

### 1.4 Logging Setup

**Files to create:**
- `core/logging_config.py` - Configure Python logging

**Effort:** 1 hour  
**Impact:** ⭐⭐ (Debugging, professionalism)

**Steps:**
1. Replace all `print()` statements with `logger.info()`, `logger.debug()`
2. Create logging module with file + console output
3. Add verbosity control to config
4. Test logging to file

---

## Phase 2: CONFIGURATION MANAGEMENT (1-2 weeks)

### 2.1 Create ConfigManager Class

**File:** `config/ConfigManager.py`

```python
class ConfigManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_and_validate()
    
    def _load_and_validate(self):
        # Load JSON
        # Validate against schema
        # Provide defaults
        # Return validated config
        pass
    
    def get(self, section, key, default=None):
        # Typed access to config
        pass
    
    def to_dict(self):
        # Return config as dict
        pass
```

**Effort:** 2-3 hours  
**Impact:** ⭐⭐⭐ (All subsequent phases depend on this)

**Integration points:**
- `main.py` - Load config once at startup
- `orchestrator.py` - Pass config object to modules
- All technique files - Read from config instead of modifying it

---

### 2.2 Stop Runtime Config Modifications

**Current problem:**
- Saber_tools, lemni_tools, pinning_RL_tools all call `update_configs()`
- Config file is written to during simulation

**Solution:**
- Techniques read config at initialization
- Never modify config.json during runtime
- Log actual parameters used to metadata file

**Effort:** 2-3 hours  
**Impact:** ⭐⭐⭐ (Reproducibility, debugging)

**Files to modify:**
- `planner/techniques/*.py` - Remove `update_configs()` calls
- `learner/CALA_control.py` - Remove config updates
- `learner/QL_learning_lattice.py` - Remove config updates

---

### 2.3 Centralize Technique Configuration

**Create:** `config/technique_configs/`
- `saber.json` - Default saber parameters
- `lemni.json` - Default lemni parameters
- `pinning.json` - Default pinning parameters
- etc.

**Effort:** 1-2 hours  
**Impact:** ⭐⭐ (Parameter management)

**Structure:**
```json
{
  "technique": "lemni",
  "parameters": {
    "c1_d": 1.0,
    "c2_d": 2.0,
    "learning": {
      "enabled": true,
      "algorithm": "CALA",
      "axes": "xz"
    }
  }
}
```

---

## Phase 3: OBJECT-ORIENTED ARCHITECTURE (2-3 weeks)

### 3.1 Create Base Classes

**File:** `planner/TechniqueBase.py`

```python
class PlanningTechnique:
    """Base class for all planning techniques"""
    
    def __init__(self, config, num_agents, dimensions):
        self.config = config
        self.num_agents = num_agents
        self.dimensions = dimensions
        self._validate_config()
    
    def _validate_config(self):
        """Override in subclass"""
        pass
    
    def compute_trajectory(self, agents_state, targets_state, time):
        """Return desired trajectory (positions/velocities)"""
        raise NotImplementedError
    
    def compute_command(self, agents_state, targets_state, time, **kwargs):
        """Return control commands for agents"""
        raise NotImplementedError
    
    @property
    def name(self):
        """Technique name"""
        raise NotImplementedError
```

**File:** `learner/LearnerBase.py`

```python
class Learner:
    """Base class for all learners"""
    
    def __init__(self, config, num_agents):
        self.config = config
        self.num_agents = num_agents
    
    def compute_action(self, state, **kwargs):
        """Return action/parameter to learn"""
        raise NotImplementedError
    
    def update(self, reward, **kwargs):
        """Update policy based on reward"""
        raise NotImplementedError
    
    def get_state(self):
        """Return learner state for logging"""
        raise NotImplementedError
```

**Effort:** 3-4 hours  
**Impact:** ⭐⭐⭐⭐ (Foundation for all technique refactoring)

---

### 3.2 Refactor Techniques to Classes

**For each technique, create a class:**

**File:** `planner/techniques/Saber.py`

```python
from planner.TechniqueBase import PlanningTechnique

class Saber(PlanningTechnique):
    """Olfati-Saber flocking"""
    
    def __init__(self, config, num_agents, dimensions):
        super().__init__(config, num_agents, dimensions)
        self.c1_a = config.get('saber', 'c1_a', 1.0)
        self.c2_a = config.get('saber', 'c2_a', 2.0)
        # ... other params
    
    def compute_command(self, agents_state, targets_state, time, **kwargs):
        # Extract logic from saber_tools.compute_cmd()
        cmd = np.zeros((3, self.num_agents))
        # ... computation
        return cmd
    
    @property
    def name(self):
        return "Saber"
```

**Repeat for:** Lemni, Reynolds, Encirclement, Cao, Starling, Shepherding, Pinning

**Effort:** 6-8 hours  
**Impact:** ⭐⭐⭐⭐ (Consistent interfaces, easier testing)

**Files to delete after:**
- `planner/techniques/saber_tools.py` (move functions to Saber class)
- `planner/techniques/encirclement_tools.py` (move to Encirclement class)
- etc.

---

### 3.3 Refactor Learners to Classes

**File:** `learner/CALA.py` (refactored from CALA_control.py)

```python
from learner.LearnerBase import Learner

class CALA(Learner):
    """Continuous Action Learning Automata"""
    
    def __init__(self, config, num_agents):
        super().__init__(config, num_agents)
        self.num_states = config.get('cala', 'num_states', num_agents * 2)
        self.learning_rate = config.get('cala', 'learning_rate', 0.5)
        self.means = np.random.uniform(...)
        self.variances = np.full(...)
    
    def compute_action(self, state, **kwargs):
        # Sample from policy
        return actions
    
    def update(self, reward, **kwargs):
        # Update means and variances based on reward
        pass
    
    def get_state(self):
        return {
            'means': self.means.copy(),
            'variances': self.variances.copy()
        }
```

**Refactor:** CALA, QLearning, ConsensusLattice, GradientEstimator

**Effort:** 4-5 hours  
**Impact:** ⭐⭐⭐ (Cleaner learning code)

---

### 3.4 Create TechniqueRegistry

**File:** `planner/TechniqueRegistry.py`

```python
class TechniqueRegistry:
    _techniques = {}
    
    @classmethod
    def register(cls, name, technique_class):
        cls._techniques[name] = technique_class
    
    @classmethod
    def get(cls, name):
        return cls._techniques[name]
    
    @classmethod
    def list(cls):
        return list(cls._techniques.keys())

# Auto-register
TechniqueRegistry.register('saber', Saber)
TechniqueRegistry.register('lemni', Lemniscate)
# ... etc
```

**Update orchestrator.py:**
```python
# OLD: if strategy == 'saber': import saber_tools
#      if strategy == 'lemni': import lemni_tools
# NEW:
technique_class = TechniqueRegistry.get(strategy)
Technique = technique_class(config, num_agents, dimensions)
```

**Effort:** 1-2 hours  
**Impact:** ⭐⭐⭐⭐ (Runtime technique selection, no more if/elif)

---

## Phase 4: DECOUPLE LEARNING FROM PLANNING (1-2 weeks)

### 4.1 Extract Reward Functions

**Problem:** Learning logic is baked into techniques
- Lemni_tools calls CALA directly
- Pinning_RL_tools calls QL directly

**Solution:** Create reward computation interface

**File:** `learner/rewards/RewardFunction.py`

```python
class RewardFunction:
    def compute(self, agents_state, targets_state, agent_actions, time):
        """Return scalar reward"""
        raise NotImplementedError

class TargetAlignmentReward(RewardFunction):
    """For lemniscate learning"""
    def compute(self, agents_state, targets_state, agent_actions, time):
        # Compute alignment to target
        return reward

class ConnectivityReward(RewardFunction):
    """For pinning learning"""
    def compute(self, agents_state, targets_state, agent_actions, time):
        # Compute k-connectivity
        return reward
```

**Effort:** 2-3 hours  
**Impact:** ⭐⭐⭐ (Decoupled learning)

---

### 4.2 Create Learner Integration Adapters

**Problem:** Each technique needs its own way of talking to learners

**Solution:** Create technique-specific learner interfaces

**File:** `learner/integrations/LemniLearner.py`

```python
class LemniLearner:
    """Adapter: connects Lemniscate technique to CALA learner"""
    
    def __init__(self, num_agents, config):
        self.learner = CALA(config, num_agents)
        self.reward_fn = TargetAlignmentReward(config)
    
    def get_actions(self, state):
        return self.learner.compute_action(state)
    
    def update(self, agents_state, targets_state, agent_actions, time):
        reward = self.reward_fn.compute(agents_state, targets_state, agent_actions, time)
        self.learner.update(reward)
```

**File:** `learner/integrations/PinningLearner.py`

```python
class PinningLearner:
    """Adapter: connects Pinning technique to Q-learning"""
    
    def __init__(self, num_agents, config):
        self.learner = QLearning(config, num_agents)
        self.reward_fn = ConnectivityReward(config)
    
    def get_actions(self, state):
        return self.learner.compute_action(state)
    
    def update(self, agents_state, targets_state, agent_actions, time):
        reward = self.reward_fn.compute(agents_state, targets_state, agent_actions, time)
        self.learner.update(reward)
```

**Update Lemniscate technique:**
```python
# OLD: from learner import CALA_control
#      lemni_learner = CALA_control.CALA(...)
#      actions = lemni_learner.sample()

# NEW:
self.learner = LemniLearner(num_agents, config)
actions = self.learner.get_actions(state)
```

**Effort:** 2-3 hours  
**Impact:** ⭐⭐⭐⭐ (Clean decoupling)

---

### 4.3 Standardize RL Interface (Critical Architecture Fix)

**Problem Identified:**
Currently CALA and Q-Learning are two very similar RL algorithms but implemented with **incompatible patterns**:

| Aspect | CALA (Lemni) | Q-Learning (Pinning) |
|--------|-------------|-------------------|
| **Integration Point** | Trajectory generation (lemni_tools.py) | Command computation (pinning_RL_tools.py) |
| **Parameter Format** | Dict with angle offsets in radians | Numpy arrays with scale factors |
| **Reward Computation** | Internal (learner computes reward) | External (passed as kwarg) |
| **State Representation** | Unstructured dict | Unstructured numpy |
| **Interface** | `get_actions()` → dict | `update_reward()` + `scale_actions()` |

**Consequence:** Cannot swap algorithms, add new learners, or share patterns. Each new RL technique requires reimplementing entire integration.

**Solution: Create Standardized RL Framework**

**File:** `learner/core/RLInterface.py`

```python
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np

@dataclass
class RLState:
    """Standardized state representation for all RL algorithms"""
    agent_positions: np.ndarray          # (n_agents, 3)
    agent_velocities: np.ndarray         # (n_agents, 3)
    target_positions: np.ndarray         # (n_targets, 3)
    obstacle_positions: np.ndarray       # (n_obstacles, 3)
    agent_graph: np.ndarray              # (n_agents, n_agents) adjacency
    time: float
    dt: float
    episode_num: int
    step_in_episode: int
    metadata: Dict[str, Any]             # Technique-specific data

class RLAlgorithm(ABC):
    """Base class for all RL algorithms (replaces CALA, QLearning, etc.)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.episode_num = 0
        self.step_count = 0
    
    @abstractmethod
    def compute_action(self, state: RLState) -> np.ndarray:
        """
        Compute action given current state.
        
        Args:
            state: RLState object with standardized representation
            
        Returns:
            Action vector (interpretation depends on technique)
        """
        pass
    
    @abstractmethod
    def update(self, reward: float, next_state: RLState, done: bool):
        """
        Update learning from reward signal.
        
        Args:
            reward: Scalar reward signal
            next_state: Next state after action
            done: Episode termination flag
        """
        pass
    
    @abstractmethod
    def reset_episode(self):
        """Reset learning state for new episode"""
        pass
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Return learnable parameters for saving/loading"""
        return {}
    
    def set_state_dict(self, state: Dict[str, Any]):
        """Restore learnable parameters"""
        pass

class RewardFunction(ABC):
    """Standardized reward computation interface"""
    
    @abstractmethod
    def compute(self, state: RLState, action: np.ndarray, 
                reward_data: Dict[str, Any]) -> float:
        """
        Compute reward for current transition.
        
        Args:
            state: Current RLState
            action: Action taken
            reward_data: Technique-specific reward signals
                        (e.g., {'k_connectivity': 0.85, 'goal_distance': 1.2})
        
        Returns:
            Scalar reward value
        """
        pass
```

**File:** `learner/core/RLAlgorithmImplementations.py`

```python
class CALAAlgorithm(RLAlgorithm):
    """CALA refactored to use standardized RLAlgorithm interface"""
    
    def __init__(self, config: Dict[str, Any], num_agents: int):
        super().__init__(config)
        self.num_agents = num_agents
        
        # CALA-specific components
        self.policy_means = np.zeros((num_agents, config.get('num_actions', 2)))
        self.policy_vars = np.ones((num_agents, config.get('num_actions', 2))) * \
                          config.get('initial_variance', 0.5)
        self.learning_rate = config.get('learning_rate', 0.01)
        self.comm = AgentCommunication(config)
        
        # History for computing reward increments
        self.prev_reward = 0.0
    
    def compute_action(self, state: RLState) -> np.ndarray:
        """Sample actions from policy using means and variances"""
        actions = np.zeros((self.num_agents, self.config.get('num_actions', 2)))
        for i in range(self.num_agents):
            actions[i] = np.random.normal(
                self.policy_means[i],
                np.sqrt(self.policy_vars[i])
            )
        return actions
    
    def update(self, reward: float, next_state: RLState, done: bool):
        """Update policy based on reward signal"""
        reward_increment = reward - self.prev_reward
        
        # Share reward with neighbors
        neighbor_rewards = self.comm.gather_from_neighbors(reward_increment)
        
        # Update means (gradient ascent)
        # This is simplified; actual CALA uses natural gradient
        self.policy_means += self.learning_rate * reward_increment
        
        self.prev_reward = reward
        self.step_count += 1
    
    def reset_episode(self):
        """Reset for new episode"""
        self.prev_reward = 0.0

class QLearningAlgorithm(RLAlgorithm):
    """Q-Learning refactored to use standardized RLAlgorithm interface"""
    
    def __init__(self, config: Dict[str, Any], num_agents: int):
        super().__init__(config)
        self.num_agents = num_agents
        
        # Q-Learning specific
        self.state_discretization = config.get('state_discretization', 10)
        self.num_actions = config.get('num_actions', 4)
        self.q_table = {}  # state_hash -> action_values
        self.learning_rate = config.get('learning_rate', 0.1)
        self.discount = config.get('discount_factor', 0.95)
        self.epsilon = config.get('epsilon_start', 0.3)
    
    def _discretize_state(self, state: RLState) -> str:
        """Convert continuous state to discrete representation"""
        # Simplified: could use k-means clustering or fixed bins
        discretized = np.round(state.agent_positions / self.state_discretization)
        return hash(tuple(discretized.flatten()))
    
    def compute_action(self, state: RLState) -> np.ndarray:
        """Epsilon-greedy action selection"""
        state_hash = self._discretize_state(state)
        
        if np.random.random() < self.epsilon:
            # Explore: random action
            actions = np.random.randn(self.num_agents, 2)
        else:
            # Exploit: best action from Q-table
            if state_hash not in self.q_table:
                self.q_table[state_hash] = np.zeros((self.num_agents, self.num_actions))
            
            q_values = self.q_table[state_hash]
            best_actions = np.argmax(q_values, axis=1)
            actions = self._convert_action_indices_to_values(best_actions)
        
        return actions
    
    def update(self, reward: float, next_state: RLState, done: bool):
        """Q-Learning update rule"""
        # Implementation details omitted for brevity
        self.step_count += 1
        # Decay exploration
        self.epsilon = max(0.01, self.epsilon * 0.995)
    
    def reset_episode(self):
        """Reset for new episode"""
        pass

class TargetAlignmentReward(RewardFunction):
    """Reward function for Lemniscate trajectory learning"""
    
    def compute(self, state: RLState, action: np.ndarray, 
                reward_data: Dict[str, Any]) -> float:
        """
        Reward agents for staying aligned with target trajectory.
        
        Returns:
            Scalar in range [-1, 1]
        """
        # Distance to targets
        distances = np.linalg.norm(
            state.target_positions - state.agent_positions.mean(axis=0)
        )
        alignment = np.exp(-distances / state.metadata.get('trajectory_scale', 1.0))
        
        # Velocity alignment
        velocity_magnitude = np.linalg.norm(state.agent_velocities, axis=1).mean()
        velocity_reward = 1.0 if velocity_magnitude > 0.1 else 0.0
        
        return 0.7 * alignment + 0.3 * velocity_reward

class ConnectivityReward(RewardFunction):
    """Reward function for Pinning control learning"""
    
    def compute(self, state: RLState, action: np.ndarray, 
                reward_data: Dict[str, Any]) -> float:
        """
        Reward agents for maintaining k-connectivity.
        
        Returns:
            Scalar in range [0, 1]
        """
        # Extract from reward_data (computed by technique)
        k_connectivity = reward_data.get('k_connectivity', 0.0)
        
        # Encourage high connectivity
        connectivity_reward = k_connectivity
        
        # Small penalty for large actions (energy efficiency)
        action_norm = np.linalg.norm(action)
        efficiency_penalty = 0.05 * min(action_norm / 10.0, 1.0)
        
        return max(0.0, connectivity_reward - efficiency_penalty)
```

**File:** `learner/core/TechniqueRLAdapter.py`

```python
class TechniqueRLAdapter:
    """
    Generic adapter connecting any planning technique to any RL algorithm.
    Eliminates need to hard-code learner integration into technique code.
    """
    
    def __init__(self, technique_name: str, learner: RLAlgorithm, 
                 reward_fn: RewardFunction, config: Dict[str, Any]):
        self.technique_name = technique_name
        self.learner = learner
        self.reward_fn = reward_fn
        self.config = config
        self.prev_action = None
    
    def compute_trajectory(self, agents, targets, obstacles, time, dt):
        """
        Called from planning technique to get learning-adjusted trajectory.
        
        Returns trajectory with learning-computed adjustments applied.
        """
        # Build standardized state
        state = RLState(
            agent_positions=agents.state[:, :3],
            agent_velocities=agents.state[:, 3:6],
            target_positions=targets.state,
            obstacle_positions=obstacles.state,
            agent_graph=agents.adjacency_matrix,
            time=time,
            dt=dt,
            episode_num=self.learner.episode_num,
            step_in_episode=self.learner.step_count,
            metadata={
                'technique': self.technique_name,
                'trajectory_scale': self.config.get('trajectory_scale', 1.0)
            }
        )
        
        # Get learning signal
        action = self.learner.compute_action(state)
        self.prev_action = action
        
        return action
    
    def update_from_reward(self, agents, targets, obstacles, action, time, dt):
        """
        Called after trajectory execution to provide reward and update learner.
        
        Args:
            action: The action taken (from compute_trajectory)
            [other state]: For reward computation
        """
        # Rebuild state for reward computation
        state = RLState(
            agent_positions=agents.state[:, :3],
            agent_velocities=agents.state[:, 3:6],
            target_positions=targets.state,
            obstacle_positions=obstacles.state,
            agent_graph=agents.adjacency_matrix,
            time=time,
            dt=dt,
            episode_num=self.learner.episode_num,
            step_in_episode=self.learner.step_count,
            metadata={'technique': self.technique_name}
        )
        
        # Compute reward (technique-specific signals passed in)
        reward_data = self._gather_technique_signals(agents, targets)
        reward = self.reward_fn.compute(state, action, reward_data)
        
        # Update learner
        done = self._check_episode_done(agents, targets)
        self.learner.update(reward, state, done)
        
        if done:
            self.learner.reset_episode()
            self.learner.episode_num += 1
    
    def _gather_technique_signals(self, agents, targets) -> Dict[str, Any]:
        """Gather technique-specific reward signals"""
        signals = {}
        
        if self.technique_name == 'pinning_RL':
            # k-connectivity metric
            signals['k_connectivity'] = self._compute_k_connectivity(agents)
        
        elif self.technique_name == 'lemni_RL':
            # Target alignment metric
            signals['target_distance'] = self._compute_target_distance(agents, targets)
        
        return signals
    
    def _compute_k_connectivity(self, agents) -> float:
        """Compute k-connectivity metric [0, 1]"""
        # Simplified
        num_edges = np.sum(agents.adjacency_matrix) / 2
        max_edges = agents.num_agents * (agents.num_agents - 1) / 2
        return min(1.0, num_edges / max_edges) if max_edges > 0 else 0.0
    
    def _compute_target_distance(self, agents, targets) -> float:
        """Compute distance to targets"""
        if len(targets.state) == 0:
            return 0.0
        dist = np.min(np.linalg.norm(
            targets.state[:, None, :] - agents.state[None, :, :3],
            axis=2
        ))
        return dist
    
    def _check_episode_done(self, agents, targets) -> bool:
        """Check if episode should terminate"""
        # Could be based on time limit, goal achievement, etc.
        return False  # Simplified
```

**Integration Example - Updated Lemni with RL:**

```python
# In lemniscate_tools.py:

def lemniscate_controller(agents, targets, obstacles, time, dt, learning_adapter=None):
    """
    Lemniscate trajectory planner, optionally with learning.
    
    Args:
        learning_adapter: TechniqueRLAdapter instance (optional)
    """
    
    # Base lemniscate trajectory
    base_trajectory = compute_base_lemniscate(agents, targets, dt)
    
    # Apply learning-based adjustments if available
    if learning_adapter is not None:
        learning_action = learning_adapter.compute_trajectory(
            agents, targets, obstacles, time, dt
        )
        # learning_action is shape (n_agents, 2) with adjustment factors
        base_trajectory += learning_action.reshape(-1, 1) * 0.1  # Scale adjustment
    
    # Compute commands from trajectory
    commands = trajectory_to_commands(agents, base_trajectory, dt)
    
    # If learning, trigger update
    if learning_adapter is not None:
        learning_adapter.update_from_reward(
            agents, targets, obstacles, learning_action, time, dt
        )
    
    return commands
```

**Integration Example - Updated Pinning with RL:**

```python
# In pinning_RL_tools.py:

def pinning_RL_controller(agents, targets, time, dt, learning_adapter=None):
    """
    Pinning control with optional reinforcement learning.
    
    Args:
        learning_adapter: TechniqueRLAdapter instance (optional)
    """
    
    # Compute base pinning commands
    commands = compute_pinning_commands(agents, targets, dt)
    
    # Apply learning-based scaling if available
    if learning_adapter is not None:
        learning_action = learning_adapter.compute_trajectory(
            agents, targets, obstacles, time, dt
        )
        # learning_action is shape (n_agents, 2) with scale factors
        commands *= (1.0 + learning_action * 0.05)  # Scale adjustment
    
    # If learning, trigger update
    if learning_adapter is not None:
        learning_adapter.update_from_reward(
            agents, targets, obstacles, learning_action, time, dt
        )
    
    return commands
```

**Usage in main.py:**

```python
# Initialize learner and adapter once
if config['technique'] == 'lemni_RL':
    learner = CALAAlgorithm(config['learner'], config['num_agents'])
    reward_fn = TargetAlignmentReward()
    learning_adapter = TechniqueRLAdapter('lemni_RL', learner, reward_fn, config)
elif config['technique'] == 'pinning_RL':
    learner = QLearningAlgorithm(config['learner'], config['num_agents'])
    reward_fn = ConnectivityReward()
    learning_adapter = TechniqueRLAdapter('pinning_RL', learner, reward_fn, config)
else:
    learning_adapter = None

# Main simulation loop
for time in simulation_times:
    trajectory = technique(agents, targets, obstacles, time, dt, learning_adapter)
    commands = trajectory_to_commands(agents, trajectory, dt)
    agents.update(commands, dt)
    # ...
```

**Benefits of Standardized Framework:**

1. **Interchangeable Algorithms:** CALA, QLearning, Gradient, Consensus can now be swapped trivially
2. **Consistent Interface:** All learners have `compute_action()` and `update()`
3. **Clean Separation:** Learning logic fully decoupled from technique code
4. **Extensible:** Adding new algorithm requires only implementing RLAlgorithm base class
5. **Testable:** Each component (algorithm, reward function, adapter) independently testable
6. **Reusable Rewards:** `TargetAlignmentReward` can be used with any learner+technique
7. **Standardized State:** All algorithms understand same RLState representation

**Migration Path:**

1. Create `learner/core/` module with base classes and implementations
2. Refactor CALA → CALAAlgorithm + TargetAlignmentReward
3. Refactor QLearning → QLearningAlgorithm + ConnectivityReward
4. Create adapters for Lemni, Pinning techniques
5. Update orchestrator to instantiate adapters
6. Gradually refactor other techniques to use same pattern
7. Remove old hardcoded learner integration from technique files

**Effort:** 5-6 hours (includes refactoring both CALA and QLearning)  
**Impact:** ⭐⭐⭐⭐⭐ (Enables pluggable learning, unblocks future learners)  
**Risk:** Medium (refactor core learning systems, needs testing)

---

## Phase 5: MODULARIZE LARGE FILES (1-2 weeks)

### 5.1 Break Down CALA (1028 lines)

**Current structure:**
- Policy representation (means, variances)
- Sampling logic
- Update logic
- Reward computation
- Neighbor communication
- Debug code
- Test code

**Split into:**

**File:** `learner/cala/CALAPolicy.py`
```python
class CALAPolicy:
    def __init__(self, num_states, action_min, action_max):
        self.means = ...
        self.variances = ...
    
    def sample(self):
        return actions
    
    def update(self, reward):
        # Update means/variances
        pass
```

**File:** `learner/cala/CALAConfig.py`
```python
class CALAConfig:
    def __init__(self, config_dict):
        self.learning_rate = config_dict['learning_rate']
        self.variance_init = config_dict['variance_init']
        # ... validate all params
```

**File:** `learner/cala/RewardComputation.py`
```python
def compute_reward(state, action, next_state, config):
    # Reward logic extracted from update_reward_increment()
    pass
```

**File:** `learner/cala/Communication.py`
```python
class AgentCommunication:
    def negotiate_with_neighbours(self, agents_states, policy_mean):
        # Leader-follower or consensus
        pass
```

**New:** `learner/CALA.py` (simple wrapper)
```python
from learner.cala import CALAPolicy, CALAConfig, RewardComputation, Communication

class CALA(Learner):
    def __init__(self, config, num_agents):
        self.config = CALAConfig(config)
        self.policy = CALAPolicy(num_states, config.action_min, config.action_max)
        self.comm = AgentCommunication(config)
    
    def compute_action(self, state, **kwargs):
        return self.policy.sample()
    
    def update(self, reward, **kwargs):
        r = RewardComputation.compute(...)
        self.policy.update(r)
```

**Effort:** 4-5 hours  
**Impact:** ⭐⭐⭐⭐ (Much more maintainable)

---

### 5.2 Extract Orchestrator Logic

**Current:** 454 lines, too many responsibilities

**Split into:**

**File:** `core/SystemBuilder.py`
```python
class SystemBuilder:
    def build(self, config):
        agents = self._build_agents(config)
        targets = self._build_targets(config)
        obstacles = self._build_obstacles(config)
        trajectory = self._build_trajectory(config)
        learners = self._build_learners(config)
        return agents, targets, obstacles, trajectory, learners
```

**File:** `core/SimulationController.py`
```python
class SimulationController:
    def __init__(self, agents, targets, obstacles, technique, learners):
        self.agents = agents
        self.technique = technique
        self.learners = learners
        self.graph = SwarmGraph(agents.state)
    
    def step(self, time, dt):
        # Compute trajectory
        # Compute commands
        # Update learning
        # Update graph
        pass
```

**New orchestrator.py:** Just factory functions that call SystemBuilder

**Effort:** 2-3 hours  
**Impact:** ⭐⭐⭐ (Clearer responsibilities)

---

## Phase 6: DATA MANAGEMENT IMPROVEMENTS (1-2 weeks)

### 6.1 Create Recording Framework

**Problem:** History object stores everything in memory (memory hog)

**Solution:** Pluggable recorder system

**File:** `io/RecorderBase.py`

```python
class Recorder:
    def record_step(self, time, agents, targets, obstacles, controller, trajectory):
        raise NotImplementedError
    
    def finalize(self):
        """Called when simulation ends"""
        pass
    
    def get_results(self):
        """Return recorded data"""
        pass
```

**File:** `io/FullStateRecorder.py`
```python
class FullStateRecorder(Recorder):
    """Record everything (current behavior)"""
    def __init__(self):
        self.history = History()
    
    def record_step(self, time, agents, targets, obstacles, controller, trajectory):
        self.history.update(...)
```

**File:** `io/MetricsOnlyRecorder.py`
```python
class MetricsOnlyRecorder(Recorder):
    """Record only connectivity, order metrics (much smaller)"""
    def __init__(self):
        self.metrics = []
    
    def record_step(self, time, agents, targets, obstacles, controller, trajectory):
        self.metrics.append({
            't': time,
            'order': compute_order(agents.state),
            'connectivity': compute_connectivity(agents.state)
        })
```

**File:** `io/StreamingRecorder.py`
```python
class StreamingRecorder(Recorder):
    """Write to disk incrementally (no memory buffer)"""
    def __init__(self, filepath):
        self.file = h5py.File(filepath, 'w')
    
    def record_step(self, time, agents, targets, obstacles, controller, trajectory):
        # Append to HDF5 without buffering
        pass
```

**Update main.py:**
```python
# OLD: Database = data_manager.History(...)

# NEW:
recorders = [
    MetricsOnlyRecorder(),  # Small, fast
    StreamingRecorder('data.h5')  # Disk I/O
]

for timestep in simulation:
    # ... physics ...
    for recorder in recorders:
        recorder.record_step(time, agents, targets, obstacles, controller, trajectory)
```

**Effort:** 3-4 hours  
**Impact:** ⭐⭐⭐ (Memory efficiency, flexibility)

---

### 6.2 Add Data Metadata

**File:** `io/DataMetadata.py`

```python
class SimulationMetadata:
    def __init__(self, config, git_hash, timestamp):
        self.config = config
        self.git_hash = git_hash
        self.timestamp = timestamp
        self.duration_seconds = None
        self.duration_timesteps = None
        self.num_agents = None
    
    def to_dict(self):
        return {
            'config': self.config.to_dict(),
            'git_hash': self.git_hash,
            'timestamp': self.timestamp,
            'duration_seconds': self.duration_seconds,
            'duration_timesteps': self.duration_timesteps,
            'num_agents': self.num_agents
        }
```

**Update recording:**
```python
metadata = SimulationMetadata(config, get_git_hash(), datetime.now())
# ... run simulation ...
metadata.duration_seconds = (datetime.now() - metadata.timestamp).total_seconds()
metadata.duration_timesteps = len(recorder.metrics)
metadata.save('metadata.json')
```

**Effort:** 1-2 hours  
**Impact:** ⭐⭐⭐ (Reproducibility)

---

## Phase 7: TESTING & VALIDATION (2-3 weeks)

### 7.1 Create Test Infrastructure

**Directory:** `tests/`

**File:** `tests/conftest.py`
```python
import pytest
import numpy as np

@pytest.fixture
def sample_config():
    return {
        'simulation': {'Ti': 0, 'Tf': 1, 'Ts': 0.1},
        'agents': {'nAgents': 5},
        'saber': {'c1_a': 1.0, 'c2_a': 2.0}
    }

@pytest.fixture
def sample_agents(sample_config):
    from agents.agents import Agents
    return Agents('saber', 3)
```

**Effort:** 1-2 hours  
**Impact:** ⭐⭐⭐ (Foundation for testing)

---

### 7.2 Unit Tests for Core Modules

**File:** `tests/unit/test_config_manager.py`
```python
def test_config_loads(sample_config):
    manager = ConfigManager(sample_config)
    assert manager.get('agents', 'nAgents') == 5

def test_config_validation():
    # Missing required param should raise
    with pytest.raises(ValueError):
        ConfigManager({'simulation': {}})
```

**File:** `tests/unit/test_agents.py`
```python
def test_agents_initialization(sample_agents):
    assert sample_agents.nAgents == 5
    assert sample_agents.state.shape == (6, 5)

def test_agents_evolve(sample_agents):
    cmd = np.random.randn(3, 5)
    sample_agents.evolve(cmd, None, 0, 0.1)
    # State should have changed
    assert not np.allclose(sample_agents.state, 0)
```

**File:** `tests/unit/test_techniques.py`
```python
def test_saber_creates(sample_config):
    from planner.techniques.Saber import Saber
    tech = Saber(sample_config, 5, 3)
    assert tech.name == "Saber"

def test_saber_computes_commands(sample_agents):
    from planner.techniques.Saber import Saber
    tech = Saber(sample_config, sample_agents.nAgents, 3)
    cmd = tech.compute_command(sample_agents.state, targets, 0)
    assert cmd.shape == (3, sample_agents.nAgents)
```

**Effort:** 4-5 hours  
**Impact:** ⭐⭐⭐⭐ (Safe refactoring)

---

### 7.3 Integration Tests

**File:** `tests/integration/test_simulation_pipeline.py`
```python
def test_full_simulation_runs(sample_config):
    # Run small sim to completion
    from core.SimulationPipeline import SimulationPipeline
    
    config = sample_config.copy()
    config['simulation']['Tf'] = 0.5  # Short
    config['agents']['nAgents'] = 3   # Small
    
    pipeline = SimulationPipeline(config)
    pipeline.initialize()
    pipeline.run()
    
    # Check data was recorded
    assert len(pipeline.recorder.metrics) > 0
```

**File:** `tests/integration/test_each_technique.py`
```python
@pytest.mark.parametrize('technique', ['saber', 'lemni', 'pinning', ...])
def test_technique_runs_to_completion(technique):
    config = get_test_config(technique)
    pipeline = SimulationPipeline(config)
    pipeline.initialize()
    pipeline.run()
    assert pipeline.completed_successfully
```

**Effort:** 2-3 hours  
**Impact:** ⭐⭐⭐⭐ (Catches regressions)

---

### 7.4 Type Hints & Type Checking

**Add to all files:**
```python
from typing import Dict, List, Tuple, Optional
import numpy as np

def compute_command(self, 
                   agents_state: np.ndarray,  # shape (6, nAgents)
                   targets_state: np.ndarray,  # shape (3, nTargets)
                   time: float) -> np.ndarray:  # returns shape (3, nAgents)
    """Compute control commands."""
    pass
```

**Setup mypy:**
```
# mypy.ini
[mypy]
python_version = 3.10
strict = True
warn_return_any = True
```

**Effort:** 3-4 hours  
**Impact:** ⭐⭐⭐ (Fewer bugs, better IDE support)

---

## Phase 8: DOCUMENTATION (2-3 weeks)

### 8.1 Code Documentation

**Update all modules with docstrings:**
```python
"""
learner/CALA.py

Continuous Action Learning Automata (CALA) implementation for decentralized swarm control.

This module provides an adaptive learning approach where each agent independently learns
action parameters based on local reward signals. Agents can communicate to reach consensus
on policy parameters.

Classes:
    CALA: Main learner class

Example:
    >>> config = {'cala': {'learning_rate': 0.5, 'num_states': 26}}
    >>> learner = CALA(config, num_agents=13)
    >>> actions = learner.compute_action(state)
    >>> learner.update(reward)
"""
```

**Effort:** 4-5 hours  
**Impact:** ⭐⭐⭐ (Onboarding, maintainability)

---

### 8.2 Architecture Documentation

**Create:** `docs/ARCHITECTURE.md`
```markdown
# Architecture Overview

## High-Level Design

The simulator follows a pipeline architecture:

1. **Configuration** - Load and validate parameters
2. **System Building** - Create agents, targets, obstacles
3. **Simulation Loop** - Evolve agents, compute commands
4. **Recording** - Store results (pluggable)
5. **Visualization** - Plot and animate results

## Module Organization

### Core Modules
- `agents/` - Agent dynamics and state
- `planner/` - Planning techniques (8 different approaches)
- `learner/` - Learning algorithms (4 approaches)
- `data/` - Data recording and storage

### Supporting Modules
- `config/` - Configuration management
- `utils/` - Graph analysis, utilities
- `visualization/` - Plots and animations

...
```

**Create:** `docs/TECHNIQUE_GUIDE.md`
```markdown
# Planning Techniques

## Overview

The simulator includes 8 planning techniques, each extending `PlanningTechnique`:

### Saber (Olfati-Saber)
- Gradient-based flocking
- Velocity alignment
- Navigation to goal
- Obstacle avoidance

### Lemniscate
- Agents trace lemniscate curves
- Can be learned via CALA
- Supports x-z axis learning

...
```

**Create:** `docs/LEARNING_GUIDE.md`
```markdown
# Learning Algorithms

## CALA (Continuous Action Learning Automata)

Used by lemniscate technique to learn offset angles.

Policy: Gaussian distribution over action space
Update: Reward-weighted Gaussian adaptation

...
```

**Effort:** 3-4 hours  
**Impact:** ⭐⭐⭐⭐ (Professional appearance, onboarding)

---

### 8.3 API Reference

**Use Sphinx to auto-generate from docstrings:**

**File:** `docs/conf.py`
```python
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']
```

**File:** `docs/api/agents.rst`
```
agents module
=============

.. automodule:: agents.agents
   :members:
   :undoc-members:
```

**Build HTML:**
```bash
cd docs
make html
```

**Effort:** 1-2 hours  
**Impact:** ⭐⭐⭐ (Professional documentation)

---

## Phase 9: EXPERIMENT MANAGEMENT (1-2 weeks)

### 9.1 Experiment Configuration System

**File:** `experiments/ExperimentConfig.py`

```python
class ExperimentConfig:
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.base_config = None
        self.parameter_sweeps = {}
        self.num_runs = 1
    
    def add_sweep(self, param_path, values):
        """Add parameter sweep"""
        # param_path = 'saber.c1_a'
        # values = [0.5, 1.0, 1.5]
        self.parameter_sweeps[param_path] = values
    
    def generate_configs(self):
        """Generate all combinations"""
        # Returns list of configs for each combination
        pass
```

**Effort:** 2-3 hours  
**Impact:** ⭐⭐⭐ (Systematic experimentation)

---

### 9.2 Experiment Runner

**File:** `experiments/ExperimentRunner.py`

```python
class ExperimentRunner:
    def __init__(self, experiment_config):
        self.config = experiment_config
        self.results_dir = f"results/{experiment_config.name}/"
    
    def run(self):
        """Execute all experiment runs"""
        configs = self.config.generate_configs()
        
        for i, config in enumerate(configs):
            print(f"Run {i+1}/{len(configs)}")
            
            pipeline = SimulationPipeline(config)
            pipeline.initialize()
            pipeline.run()
            
            # Save results
            run_dir = f"{self.results_dir}/run_{i:03d}/"
            pipeline.save(run_dir)
            
            # Log completion
            self._log_run(i, config, pipeline)
```

**Effort:** 2-3 hours  
**Impact:** ⭐⭐⭐⭐ (Research capability)

---

### 9.3 Result Analysis

**File:** `analysis/ResultAnalyzer.py`

```python
class ResultAnalyzer:
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.results = self._load_all_runs()
    
    def compare_metrics(self, metric_name):
        """Compare metric across all runs"""
        # Returns statistics: mean, std, min, max
        pass
    
    def plot_comparison(self, metric_name, output_path):
        """Generate comparison plot"""
        pass
```

**Effort:** 2-3 hours  
**Impact:** ⭐⭐⭐ (Research analysis)

---

## Phase 10: OPTIMIZATION (1-2 weeks)

### 10.1 Profile Memory Usage

**File:** `tools/profile_memory.py`

```python
def profile_memory(config):
    import tracemalloc
    tracemalloc.start()
    
    pipeline = SimulationPipeline(config)
    pipeline.initialize()
    pipeline.run()
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current: {current / 1e6:.1f} MB")
    print(f"Peak: {peak / 1e6:.1f} MB")
```

**Effort:** 1 hour  
**Impact:** ⭐⭐⭐ (Understand memory bottlenecks)

---

### 10.2 Parallel Recording

**Problem:** SimulationController is single-threaded bottleneck

**Solution:** Decouple recording via queue

```python
from queue import Queue
from threading import Thread

record_queue = Queue()

def recorder_thread():
    while True:
        data = record_queue.get()
        if data is None:  # Sentinel
            break
        for recorder in recorders:
            recorder.record_step(**data)

# Start thread
recorder = Thread(target=recorder_thread)
recorder.daemon = True
recorder.start()

# In main loop
record_queue.put({
    'time': t,
    'agents': agents,
    'targets': targets,
    # ...
})
```

**Effort:** 2-3 hours  
**Impact:** ⭐⭐⭐ (Performance for large agents)

---

### 10.3 Vectorized Computations

**Audit for loops that could be vectorized:**
- Agent-agent distance calculations (use scipy.cdist)
- Command computations (use numpy broadcasting)
- Metric calculations

**Effort:** 2-3 hours  
**Impact:** ⭐⭐⭐ (Speed improvement)

---

## Implementation Timeline

### Week 1: Foundation (Phases 1-2)
- Setup infrastructure
- Document current config
- Clean dead code
- Setup logging
- Create ConfigManager

### Week 2-3: Architecture (Phases 3-4)
- Create base classes
- Refactor techniques to classes
- Refactor learners to classes
- Create TechniqueRegistry
- Decouple learning from planning

### Week 4: Data & Cleanup (Phases 5-6)
- Break down CALA
- Extract orchestrator logic
- Create recording framework
- Add metadata system

### Week 5-6: Testing (Phase 7)
- Create test infrastructure
- Write unit tests
- Write integration tests
- Add type hints

### Week 7: Documentation (Phase 8)
- Add docstrings
- Write guides
- Generate API docs
- Create examples

### Week 8+: Enhancement (Phases 9-10)
- Experiment management
- Result analysis
- Memory profiling
- Parallelization
- Vectorization

---

## Risk Mitigation

### Test as You Go
- After each phase, run full simulation to verify behavior is unchanged
- Keep baseline results to compare against

### Git Workflow
```bash
# Start new feature
git checkout -b phase-3-oo-refactoring

# Frequent commits
git commit -m "Create PlanningTechnique base class"
git commit -m "Refactor Saber to OO class"

# When phase complete
git commit -m "Complete Phase 3: OO refactoring [verified]"
```

### Preserve Old Code
- Keep old saber_tools.py, encirclement_tools.py as reference
- Can delete once new classes are tested and working
- Branch preserves history

### Validation Checkpoints

After each phase:
1. Run baseline simulation (known config)
2. Compare metrics to current version
3. Verify no regressions
4. Commit working state

---

## Success Criteria

After completing this roadmap:

✅ **Code Quality**
- All modules <400 lines (except data storage)
- No hardcoded parameters (all in config)
- Type hints on all public functions
- >80% test coverage

✅ **Modularity**
- Can swap techniques at runtime
- Can add new technique in <1 hour
- Can add new learner in <1 hour
- No circular dependencies

✅ **Reproducibility**
- Run ID saved with every result
- Config captured at runtime
- Git hash recorded
- Results directory structure consistent

✅ **Performance**
- Memory usage <50% reduction (with MetricsOnlyRecorder)
- Parallel recording thread (optional)
- Vectorized key computations
- Profiling tools available

✅ **Maintainability**
- New developers can understand system in 2 hours
- Architecture doc + guides
- API reference generated
- Example configs for each technique

✅ **Research Capability**
- Experiment runner for parameter sweeps
- Result analyzer for comparisons
- Easy to run multi-config studies
- Results reproducible from metadata

---

## Questions Before Starting

1. **Python version target?** (Currently 3.7+, could enforce 3.10+)
2. **Performance critical regions?** (Where does most time go?)
3. **Backwards compatibility needed?** (Can break old configs?)
4. **Research focus?** (Learning? Dynamics? New techniques?)
5. **Team size?** (Solo? Multiple contributors?)

---

**This roadmap provides a clear path from "working but messy" to "professional research tool."**  
**Each phase is self-contained and testable. Work methodically and verify after each step.**

