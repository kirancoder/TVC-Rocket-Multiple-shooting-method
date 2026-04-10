# main.py
from solver import solve_ipopt
from postprocess import extract_solution
import numpy as np

if __name__ == "__main__":
    sol, info = solve_ipopt()
    print("IPOPT done. Info:", info)
    extract_solution(sol)


# from initial_guess import build_initial_guess
# from ms_problem import constraints

# x0 = build_initial_guess()
# c0 = np.array(constraints(x0))

# con0 = np.asarray(constraints(x0), dtype=float)
# m_cons = con0.size
# print("Initial guess max continuity error = ", np.max(np.abs(con0)))