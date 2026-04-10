# initial_guess.py
# import numpy as np
# from config import K, state_dim, control_dim, x_0, z_0, vx_0, vz_0, theta_0, m0, S
# from integrator import integrate_interval
# from ms_problem import constraints
# import jax.numpy as jnp

# def rollout_error(controls, tf_guess, S_rollout):
#     """
#     Forward rollout given controls (K x 2) and tf_guess.
#     Returns flattened variable vector and max continuity error.
#     """
#     dt = tf_guess / K
#     s = np.array([x_0, z_0, vx_0, vz_0, theta_0, m0], dtype=float)
#     states = [s]
#     for i in range(K):
#         s_next = np.array(integrate_interval(s, controls[i], dt, S_rollout), dtype=float)
#         # if mass becomes nonphysical clamp it and break (we'll treat as bad)
#         if not np.isfinite(s_next).all() or s_next[5] <= 0.0:
#             return None, 1e20
#         states.append(s_next)
#         s = s_next
#     states = np.vstack(states)
#     vars_vec = np.concatenate([states.ravel(), controls.ravel(), [tf_guess]]).astype(float)

#     # compute constraint residuals (continuity + BCs) with ms_problem.constraints
#     try:
#         c = np.asarray(constraints(vars_vec), dtype=float)
#     except Exception:
#         return vars_vec, 1e20
#     max_err = float(np.max(np.abs(c)))
#     return vars_vec, max_err


# def build_initial_guess(threshold=1e-3, verbose=True):
#     """
#     Adaptive search for a near-feasible initial guess.
#     Tries a grid of (u1_guess, tf_guess) values (u1 in normalized 0..1).
#     Returns the first vars vector with max continuity error <= threshold.
#     If no candidate found, returns the best-found vector (lowest error).
#     """
#     # candidate values (start conservative; we can expand if needed)
#     u1_candidates = [0.02, 0.01, 0.005, 0.003, 0.001]   # very small normalized thrusts
#     tf_candidates = [40.0, 30.0, 20.0, 15.0]           # shorter times
#     S_rollout = max(S, 40)                             # use high substeps for rollout stability

#     best_vars = None
#     best_err = 1e20
#     best_params = None

#     for tf_guess in tf_candidates:
#         for u1_guess in u1_candidates:
#             # build controls
#             controls = np.zeros((K, control_dim), dtype=float)
#             controls[:, 0] = u1_guess
#             controls[:, 1] = 0.0
#             vars_vec, max_err = rollout_error(controls, tf_guess, S_rollout)

#             if verbose:
#                 print(f"trial u1={u1_guess:.4f}, tf={tf_guess:.1f} => max_err = {max_err:.6g}")

#             if max_err < best_err:
#                 best_err = max_err
#                 best_vars = vars_vec
#                 best_params = (u1_guess, tf_guess)

#             if max_err <= threshold:
#                 if verbose:
#                     print("Found feasible initial guess:", best_params, "err=", max_err)
#                 return best_vars

#     # if we get here nothing met threshold; return best found (still better than nothing)
#     if verbose:
#         print("No candidate met threshold. Best found:", best_params, "err=", best_err)
#     return best_vars


# initial_guess.py
import numpy as np
from config import K, state_dim, control_dim, x_0, z_0, vx_0, vz_0, theta_0, m0, g
from integrator import integrate_interval

def build_initial_guess():

    # free fall guess
    u1_guess = 0.0
    u2_guess = 0.0

    controls = np.zeros((K, control_dim))
    controls[:, 0] = u1_guess
    controls[:, 1] = u2_guess

    # free fall time guess
    tf_guess = np.sqrt(2 * z_0 / g)
    dt = tf_guess / K

    S_effective = 40

    s = np.array([x_0, z_0, vx_0, vz_0, theta_0, m0], dtype=float)
    states = [s]

    for i in range(K):
        s_next = np.array(integrate_interval(s, controls[i], dt, S_effective), float)
        states.append(s_next)
        s = s_next

    states = np.vstack(states)

    return np.concatenate([states.ravel(), controls.ravel(), [tf_guess]]).astype(float)
