import pycutest
from pysolnp import SOLNP
import numpy as np

# SOLNP+ configuration
SOLNP_CONFIG = {
    "tol": 1e-4,          # Convergence tolerance
    "tol_con": 1e-6,      # Infeasibility tolerance
    "maxfev": 1000,        # Maximum function evaluations
    "max_iter": 100        # Maximum iterations
}

def evaluate_problem(problem_name):
    """Evaluate a single problem using SOLNP+ and return results."""
    
    try:
        # Load problem
        problem = pycutest.import_problem(problem_name)

        # Get constraint information
        if problem.m > 0:
            is_eq_cons = problem.is_eq_cons
            m1 = np.sum(is_eq_cons)  # Number of equality constraints
            m2 = problem.m - m1      # Number of inequality constraints
        else:
            m1, m2 = 0, 0
        
        # Define cost function for SOLNP+
        def cost(p):
            obj_value = problem.obj(p)
            if problem.m == 0:
                return [obj_value]
            cons_value = problem.cons(p)
            eq_cons = cons_value[is_eq_cons]
            ineq_cons = cons_value[~is_eq_cons]
            return [obj_value] + eq_cons.tolist() + ineq_cons.tolist()
        
        # Set up SOLNP+ problem parameters
        prob = {
            'p0': problem.x0,
            'pbl': problem.bl if problem.bl is not None else [-np.inf] * problem.n,
            'pbu': problem.bu if problem.bu is not None else [np.inf] * problem.n,
            'ibl': problem.cl[~is_eq_cons] if m2 > 0 else [],
            'ibu': problem.cu[~is_eq_cons] if m2 > 0 else []
        }
        
        # Run SOLNP+
        mysolnp = SOLNP(prob=prob, cost=cost, op=SOLNP_CONFIG)
        info = mysolnp.run()
        
    except Exception as e:
        info = {
            "status": "error",
            "message": str(e)
        }
    
    return info

if __name__ == "__main__":
    problems = ['3PK', 'AKIVA', 'ARGAUSS', 'ARGLINA', 'ARGLINB', 'ARGLINC', 'ARGTRIGLS', 'BA-L1LS', 'BA-L1SPLS', 'BENNETT5LS', 'BROWNAL', 'CANTILVR', 'CHEBYQAD', 'CHNROSNB', 'CHNRSNBM', 'COATING', 'DANWOODLS', 'DECONVB', 'DECONVC', 'DECONVU', 'DEVGLA1', 'DEVGLA1B', 'DEVGLA2', 'DIAMON3DLS', 'DIXCHLNG', 'DMN15103LS', 'DMN15333LS', 'DUAL1', 'DUAL2', 'DUAL3', 'DUAL4', 'ERRINROS', 'ERRINRSM', 'EXTRASIM', 'FBRAIN3LS', 'GAUSS2LS', 'GENROSE', 'HADAMALS', 'HAHN1LS', 'HATFLDF', 'HATFLDFLNE', 'HIELOW', 'HIMMELP5', 'HIMMELP6', 'HOLMES', 'HS10', 'HS11', 'HS110', 'HS268', 'HS33', 'HS64', 'HS72', 'HS90', 'HS91', 'HS92', 'HUBFIT', 'HYDC20LS', 'HYDROELS', 'LEVYMONT', 'LINSPANH', 'LSQFIT', 'LUKSAN11LS', 'LUKSAN12LS', 'LUKSAN13LS', 'LUKSAN14LS', 'LUKSAN15LS', 'LUKSAN16LS', 'LUKSAN17LS', 'LUKSAN21LS', 'LUKSAN22LS', 'MANCINO', 'MISRA1CLS', 'MNISTS0LS', 'MNISTS5LS', 'OSBORNEA', 'OSCIPATH', 'PALMER2A', 'PALMER3A', 'PALMER4A', 'PALMER7A', 'PARKCH', 'PENALTY2', 'PENALTY3', 'PENLT1NE', 'PENLT2NE', 'POLAK1', 'POWELLSQ', 'PROBPENL', 'QING', 'S268', 'SENSORS', 'SPECAN', 'SPIN2LS', 'STRATEC', 'TAME', 'TFI1', 'TOINTGOR', 'TOINTPSP', 'TOINTQOR', 'VARDIM', 'VARDIMNE', 'WAYSEA1B', 'WEEDS']
    inf_count = 0
    for problem in problems:
        print(f"Running {problem}...", flush=True)
        result = evaluate_problem(problem)
        print("\n", flush=True)
        if result['obj'] == np.inf or result['obj'] == -np.inf:
            inf_count += 1
    print(f"Problem with inf: {inf_count}")
