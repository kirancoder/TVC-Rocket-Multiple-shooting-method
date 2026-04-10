# postprocess.py
import numpy as np
import pandas as pd
from config import K, state_dim, control_dim

def extract_solution(w_opt):
    # unpack variable structure
    N_states = (K+1)*state_dim
    N_controls = K*control_dim
    
    states = w_opt[:N_states].reshape((K+1, state_dim))
    controls = w_opt[N_controls+N_states - N_controls : N_states+N_controls].reshape((K, control_dim))
    tf = w_opt[-1]
    
    # time grids
    t_states = np.linspace(0, tf, K+1)      # K+1 state grid
    t_controls = np.linspace(0, tf, K, endpoint=False)  # K control intervals
    
    dt = tf / K
    
    # --------------------------
    # Save state trajectory
    # --------------------------
    df_states = pd.DataFrame({
        "t": t_states,
        "x": states[:,0],
        "z": states[:,1],
        "vx": states[:,2],
        "vz": states[:,3],
        "theta": states[:,4],
        "mass": states[:,5],
    })
    df_states.to_csv("ms_states.csv", index=False)
    print("Saved ms_states.csv")

    # --------------------------
    # Save controls per shooting interval
    # --------------------------
    df_controls = pd.DataFrame({
        "t_interval": t_controls,
        "u1": controls[:,0],
        "u2": controls[:,1],
        "dt_interval": dt,   # Same dt for all intervals (uniform)
    })
    df_controls.to_csv("ms_controls.csv", index=False)
    print("Saved ms_controls.csv")

    # --------------------------
    # Combined file (easy inspection)
    # --------------------------
    df_full = df_states.copy()
    df_full["u1"] = np.append(controls[:,0], controls[-1,0])  # repeat last for same length
    df_full["u2"] = np.append(controls[:,1], controls[-1,1])

    df_full.to_csv("ms_traj_full.csv", index=False)
    print("Saved ms_traj_full.csv")

    # return data for possible plotting
    return df_states, df_controls, df_full
