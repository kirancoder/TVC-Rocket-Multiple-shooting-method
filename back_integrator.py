from dynamics import f
from integrator import integrate_interval
import numpy as np
import pandas as pd
from config import S

df_states = pd.read_csv("ms_states.csv")
df_controls = pd.read_csv("ms_controls.csv")

states = df_states[["x","z","vx","vz","theta","mass"]].values
controls = df_controls[["u1","u2"]].values
dt = df_controls["dt_interval"].iloc[0]

# integrate forward from s_0
s = states[0]
for i in range(len(controls)):
    s = integrate_interval(s, controls[i], dt, S)
    print("interval", i, "error:", np.linalg.norm(s - states[i+1]))
