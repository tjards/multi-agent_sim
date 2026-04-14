#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 11:54:11 2024

@author: tjards

"""

#%% import stuff
# ------------
import json
import h5py
import numpy as np
from scipy import sparse
from datetime import datetime
import os

current_datetime = datetime.now()
formatted_date = current_datetime.strftime("%Y%m%d_%H%M%S")

# Memory threshold: skip dense reconstruction for n^2 fields above this (bytes)
_DENSE_SAVE_LIMIT = 2 * 1024**3  # 2 GB
#data_directory = 'data'
#file_path = os.path.join(data_directory, f"data_{formatted_date}.json")
#file_path = os.path.join(data_directory, "data.json")
#file_path = os.path.join(data_directory, "data.h5")

#%% helpers
# -------

def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj
    
# fields stored as sparse lists in History
_SPARSE_FIELDS = {'connectivity', 'pins_all', 'lattices', 'lattice_violations'}

def _reconstruct_dense_3d(sparse_list, nNodes):
    """Reconstruct a dense [nSteps, n, n] array from a list of sparse matrices."""
    nSteps = len(sparse_list)
    result = np.zeros((nSteps, nNodes, nNodes))
    for i, mat in enumerate(sparse_list):
        if mat is not None:
            if sparse.issparse(mat):
                result[i, :, :] = mat.toarray()
            else:
                result[i, :, :] = mat
    return result


def _save_sparse_hdf5(group, key, sparse_list, nNodes):
    """Save a list of sparse matrices to HDF5 in COO format with offsets."""
    all_rows = []
    all_cols = []
    all_data = []
    offsets = [0]

    for mat in sparse_list:
        if mat is not None and sparse.issparse(mat):
            coo = mat.tocoo()
            all_rows.append(coo.row.astype(np.int32))
            all_cols.append(coo.col.astype(np.int32))
            all_data.append(coo.data)
            offsets.append(offsets[-1] + len(coo.data))
        elif mat is not None:
            # dense matrix stored as sparse entry
            nz = np.nonzero(mat)
            all_rows.append(nz[0].astype(np.int32))
            all_cols.append(nz[1].astype(np.int32))
            all_data.append(mat[nz])
            offsets.append(offsets[-1] + len(nz[0]))
        else:
            offsets.append(offsets[-1])

    sparse_grp = group.create_group(key)
    sparse_grp.attrs['format'] = 'sparse_coo'
    sparse_grp.attrs['nNodes'] = nNodes
    sparse_grp.attrs['nSteps'] = len(sparse_list)

    if all_rows:
        sparse_grp.create_dataset('rows', data=np.concatenate(all_rows))
        sparse_grp.create_dataset('cols', data=np.concatenate(all_cols))
        sparse_grp.create_dataset('data', data=np.concatenate(all_data))
    else:
        sparse_grp.create_dataset('rows', data=np.array([], dtype=np.int32))
        sparse_grp.create_dataset('cols', data=np.array([], dtype=np.int32))
        sparse_grp.create_dataset('data', data=np.array([], dtype=np.float64))
    sparse_grp.create_dataset('offsets', data=np.array(offsets, dtype=np.int64))


def _load_sparse_hdf5(group, key):
    """Load sparse COO data from HDF5 and reconstruct as dense 3D array."""
    sparse_grp = group[key]
    nNodes = int(sparse_grp.attrs['nNodes'])
    nSteps = int(sparse_grp.attrs['nSteps'])
    rows = sparse_grp['rows'][:]
    cols = sparse_grp['cols'][:]
    data = sparse_grp['data'][:]
    offsets = sparse_grp['offsets'][:]

    result = np.zeros((nSteps, nNodes, nNodes))
    for i in range(nSteps):
        start, end = offsets[i], offsets[i + 1]
        if end > start:
            result[i, rows[start:end], cols[start:end]] = data[start:end]
    return result


def save_data_HDF5(data, file_path):
    
    history_data = data.__dict__
    nNodes = getattr(data, '_nNodes', 0)
    
    with h5py.File(file_path, 'w') as file:
        
        history_group = file.create_group('History')
        
        # Save data under the History group
        for key, value in history_data.items():

            # skip private attributes
            if key.startswith('_'):
                continue

            # handle sparse list fields
            if key in _SPARSE_FIELDS and isinstance(value, list):
                nSteps = len(value)
                dense_bytes = nSteps * nNodes * nNodes * 8

                if dense_bytes <= _DENSE_SAVE_LIMIT and nNodes > 0:
                    # small enough: reconstruct dense for backward compat
                    dense = _reconstruct_dense_3d(value, nNodes)
                    history_group.create_dataset(key, data=dense)
                else:
                    # too large: save in sparse COO format
                    _save_sparse_hdf5(history_group, key, value, nNodes)
                continue
            
            # convert plain lists to numpy arrays for HDF5
            if isinstance(value, list):
                try:
                    value = np.array(value)
                except (ValueError, TypeError):
                    # inhomogeneous list — skip
                    continue

            try:
                history_group.create_dataset(key, data=value)
            except (TypeError, ValueError):
                # skip non-serializable attributes (e.g. objects, scalars h5py can't handle)
                pass

def load_data_HDF5(group, key, file_path_h5):
    
    # open the HDF5 file
    with h5py.File(file_path_h5, 'r') as file:
        
        # check if group exists in the file
        if group in file:
            
            # access group
            history_group = file[group]
            
            # check if this key exists in group
            if key in history_group:

                item = history_group[key]

                # handle sparse COO format (from Phase 4 sparse storage)
                if isinstance(item, h5py.Group) and item.attrs.get('format') == 'sparse_coo':
                    values = _load_sparse_hdf5(history_group, key)
                else:
                    # standard dense dataset
                    values = item[:]
                
            else:
                print("Key not found within group.")
                values = None
        else:
            print("Group not found in the HDF5 file.")
            values = None
            
    # return the key and values
    return key, values

#%% intermediate object to store data (eventually, call this interatively)
# -----------------------------------
class History:
    
    # note: break out the Metrics stuff int another class 
    
    def __init__(self, Agents, Targets, Obstacles, Controller, Trajectory, Ts, Tf, Ti, f):
        
        nSteps = int(Tf/Ts+1)
        nAgents = Agents.nAgents
        self._nNodes = nAgents  # used by save_data_HDF5 for sparse reconstruction
        
        # initialize O(n) storage (scales linearly with agent count)
        self.t_all               = np.zeros(nSteps)
        self.states_all          = np.zeros([nSteps, len(Agents.state), nAgents])
        self.cmds_all            = np.zeros([nSteps, len(Controller.cmd), nAgents])
        self.targets_all         = np.zeros([nSteps, len(Targets.targets), nAgents])
        self.obstacles_all       = np.zeros([nSteps, len(Obstacles.obstacles), Obstacles.nObs])
        self.centroid_all        = np.zeros([nSteps, len(Agents.centroid), 1])
        self.f_all               = np.ones(nSteps)
        self.lemni_all           = np.zeros([nSteps, 2, nAgents])
        nMetrics            = 12
        self.metrics_order_all   = np.zeros((nSteps,nMetrics))
        self.metrics_order       = np.zeros((1,nMetrics))
        self.local_k_connectivity = [0]*nSteps
        self.swarm_prox = 0
        
        # initialize O(n^2) storage as sparse lists
        # each entry is a scipy.sparse.csr_matrix (or None)
        self.pins_all            = [None] * nSteps
        self.connectivity        = [None] * nSteps
        self.lattices            = [None] * nSteps
        self.lattice_violations  = [None] * nSteps
        
        # if there are quadcopters
        dynamics = Agents.dynamics_type
        if dynamics == 'quadcopter':
            self.quads_states_all   = np.zeros([nSteps, 21, nAgents])
            self.quad_w_cmd_all     = np.zeros([nSteps,4, nAgents])
            self.quads_sDes_all     = np.zeros([nSteps, 21, nAgents])

        # store the initial conditions
        self.t_all[0]                = Ti
        self.states_all[0,:,:]       = Agents.state
        self.cmds_all[0,:,:]         = Controller.cmd
        self.targets_all[0,:,:]      = Targets.targets
        self.obstacles_all[0,:,:]    = Obstacles.obstacles
        self.centroid_all[0,:,:]     = Agents.centroid
        self.f_all[0]                = f
        self.metrics_order_all[0,:]  = self.metrics_order
        self.lemni_all[0,:,:]          = Trajectory.lemni
        self.pins_all[0]             = sparse.csr_matrix(Controller.pin_matrix)
        self.connectivity[0]         = sparse.csr_matrix(Controller.Graphs.A)
        self.local_k_connectivity[0] = list(Controller.Graphs.local_k_connectivity.values())
        
        # lattice initial
        self.lattices[0]             = sparse.csr_matrix(Controller.lattice) if hasattr(Controller, 'lattice') else None
        
        # store the walls
        self.walls_plots     = Obstacles.walls_plots
        
        # lattice violations (same sparse list)
        self.lattice_violations[0]   = sparse.csr_matrix(np.zeros((nAgents, nAgents)))
        
        
    def sigma_norm(self, z): 
        
        eps = 0.5
        norm_sig = (1/eps)*(np.sqrt(1+eps*np.linalg.norm(z)**2)-1)
        return norm_sig

    def update(self, Agents, Targets, Obstacles, Controller, Trajectory, t, f, i):
        
        # core O(n) storage
        self.t_all[i]                = t
        self.states_all[i,:,:]       = Agents.state
        self.cmds_all[i,:,:]         = Controller.cmd
        self.targets_all[i,:,:]      = Targets.targets
        self.obstacles_all[i,:,:]    = Obstacles.obstacles
        self.centroid_all[i,:,:]     = Agents.centroid
        self.f_all[i]                = f
        self.lemni_all[i,:,:]          = Trajectory.lemni
        self.local_k_connectivity[i] = list(Controller.Graphs.local_k_connectivity.values())

        # sparse O(edges) storage for n×n fields
        self.pins_all[i]             = sparse.csr_matrix(Controller.pin_matrix)
        self.connectivity[i]         = sparse.csr_matrix(Controller.Graphs.A)
        self.lattices[i]             = sparse.csr_matrix(Controller.lattice)
        
        if 'consensus_lattice' in Controller.Learners:
            violations = Controller.Graphs.A * Controller.Learners['consensus_lattice'].compute_violations(Agents.state[0:3,:])
            self.lattice_violations[i] = sparse.csr_matrix(violations)
        else:
            self.lattice_violations[i] = sparse.csr_matrix((Agents.nAgents, Agents.nAgents))

        # metrics
        self.metrics_order[0,0]      = Agents.order(Agents.state[3:6,:])
        self.metrics_order[0,1:7]    = Agents.separation(Agents.state[0:3,:],Targets.targets[0:3,:],Obstacles.obstacles, Controller.Graphs_connectivity.A)
        self.metrics_order[0,7:9]    = Agents.energy(Controller.cmd)
        self.metrics_order[0,9:12]   = Agents.spacing(Agents.state[0:3,:], Controller.lattice.min())
        self.metrics_order_all[i,:]  = self.metrics_order
        self.swarm_prox              = self.sigma_norm(Agents.centroid.ravel()-Targets.targets[0:3,0])
            
        # if there are quadcopters
        dynamics = Agents.dynamics_type
        if dynamics == 'quadcopter': 
            for q in range(0,Agents.nAgents):
                self.quads_states_all[i,:,q] = Agents.quadList[q].state 
                self.quad_w_cmd_all[i,:,q]   = Agents.llctrlList[q].w_cmd
                self.quads_sDes_all[i,:,q]   = Agents.sDesList[q]
   
