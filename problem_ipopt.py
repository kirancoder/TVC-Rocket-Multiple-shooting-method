# problem_ipopt.py
import numpy as np
from jax import jacfwd, jit
from ms_problem import objective, constraints
from jax import jacfwd, jacrev

# JIT versions
obj_jit = jit(objective)
con_jit = jit(constraints)
obj_grad = jit(jacfwd(objective))  # gradient of scalar -> shape (n,)
con_jac = jit(jacfwd(constraints)) # jacobian -> shape (m, n)
obj_hess = jit(jacrev(jacfwd(objective)))  # (n,n)

class MyIpoptProblem:
    def __init__(self, n_vars, m_cons):
        self.n = n_vars
        self.m = m_cons

    # simple-name API used by cyipopt
    def objective(self, x):
        return float(obj_jit(x))

    def gradient(self, x):
        return np.asarray(obj_grad(x), dtype=float)

    def constraints(self, x):
        return np.asarray(con_jit(x), dtype=float)

    def jacobian(self, x):
        J = np.asarray(con_jac(x), dtype=float)
        # flatten row-major
        return J.ravel(order='C')

    def hessian(self, x, lagrange, obj_factor):
        H = obj_factor * np.asarray(obj_hess(x), dtype=float)
        H = 0.5*(H + H.T)
        # Return lower-triangular entries (as IPOPT expects full dense? cyipopt accepts full dense lower triangular)
        idx = np.tril_indices(self.n)
        return H[idx]
