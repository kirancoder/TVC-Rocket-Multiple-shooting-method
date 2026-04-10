# solver.py
import numpy as np
from cyipopt import Problem
from initial_guess import build_initial_guess
from problem_ipopt import MyIpoptProblem
from ms_problem import constraints
from jax import jacfwd, jit
from config import K, state_dim, control_dim, t_max, m0

def solve_ipopt():
    # 1. Initial guess
    x0 = build_initial_guess()
    n_vars = x0.size

    # 2. Evaluate constraints at x0 to determine m_cons
    con0 = np.asarray(constraints(x0), dtype=float)
    m_cons = con0.size
    print("Initial guess max continuity error = ", np.max(np.abs(con0)))

    # 3. Create sparse problem interface
    problem = MyIpoptProblem(n_vars, m_cons)

    # ---------- BOUNDS ----------
    N_states = (K+1)*state_dim
    N_controls = K*control_dim

    lb = -np.inf * np.ones(n_vars)
    ub =  np.inf * np.ones(n_vars)

    # STATE BOUNDS (recommended numerical bounds)
    lb_states = []
    ub_states = []

    for i in range(K+1):
        # x    z     vx      vz     theta       m
        lb_states.extend([-1e6, -1e6, -5e3, -5e3, -10.0,  1.0])   # mass >= 1
        ub_states.extend([ 1e6,  1e6,  5e3,  5e3,  10.0, m0])  # mass <= m0

    lb[:N_states] = lb_states
    ub[:N_states] = ub_states

    # CONTROL BOUNDS
    ctrl_start = N_states
    for i in range(K):
        lb[ctrl_start + 2*i + 0] = 0.0      # u1
        ub[ctrl_start + 2*i + 0] = 1.0

        lb[ctrl_start + 2*i + 1] = -1.0     # u2
        ub[ctrl_start + 2*i + 1] =  1.0

    # FINAL TIME BOUNDS
    lb[-1] = 1e-3
    ub[-1] = t_max

    # CONSTRAINT BOUNDS (all equalities)
    cl = np.zeros(m_cons)
    cu = np.zeros(m_cons)

    # 4. Create IPOPT problem
    nlp = Problem(n_vars, m_cons, problem, lb=lb, ub=ub, cl=cl, cu=cu)

    # 5. Set options
    nlp.add_option("max_iter", 5000)
    nlp.add_option("tol", 1e-6)
    nlp.add_option("acceptable_tol", 1e-3)
    nlp.add_option("acceptable_iter", 20)
    nlp.add_option("linear_solver", "mumps")
    nlp.add_option("mu_strategy", "adaptive")
    nlp.add_option("hessian_approximation", "limited-memory")
    nlp.add_option("nlp_scaling_method", "gradient-based")
    nlp.add_option("nlp_scaling_max_gradient", 100.0)

    # 6. Solve
    x0_np = np.asarray(x0, dtype=float)
    x_opt, info = nlp.solve(x0_np)

    return x_opt, info

if __name__ == "__main__":
    w_opt, info = solve_ipopt()
    print("Optimization info: ", info)

