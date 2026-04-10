# diagnose_continuity.py
import numpy as np
from initial_guess import build_initial_guess
from ms_problem import unpack_vars, constraints
from config import K, state_dim, control_dim
from integrator import integrate_interval

def per_interval_errors(vars_vec):
    states, controls, tf = unpack_vars(vars_vec)
    dt = tf / K
    errs = np.zeros(K)
    for i in range(K):
        s_i = states[i]
        u_i = controls[i]
        s_end = np.array(integrate_interval(s_i, u_i, dt, 40), dtype=float)
        errs[i] = np.linalg.norm(s_end - states[i+1])
    return errs

if __name__ == "__main__":
    x0 = build_initial_guess(verbose=True)
    if x0 is None:
        print("No initial guess found.")
    else:
        errs = per_interval_errors(x0)
        print("max err:", errs.max())
        # print top 8 worst intervals
        idx = np.argsort(-errs)[:8]
        for k in idx:
            print(f"interval {k:3d}: err = {errs[k]:.6g}")
