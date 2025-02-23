import nlopt
import numpy as np
import pycutest
import csv
import time
from datetime import datetime

# Configuration parameters
MAX_VARIABLES = 500       # Maximum number of variables
MAX_CONSTRAINTS = 1000    # Maximum number of constraints
RESULTS_FILE = "../results/nlopt_pycutest_with_grad_results.csv"  # Results file name
CUTEST_METADATA = "../cutest.csv"         # CUTEst metadata file
EPSILON = 1e-4            # Finite difference step size

# Optimizer configuration
OPTIMIZER_CONFIG = {
    "xtol_rel": 1e-4,
    "ftol_rel": 1e-4,
    "maxeval": 1000,
    "maxtime": 30         # Maximum optimization time per problem (seconds)
}

# NLOPT success status codes
SUCCESS_CODES = {
    nlopt.SUCCESS: "Success",
    nlopt.STOPVAL_REACHED: "Stop value reached",
    nlopt.FTOL_REACHED: "Function tolerance reached",
    nlopt.XTOL_REACHED: "Parameter tolerance reached"
}

def load_filtered_problems(csv_path):
    """Load and filter problem names from CSV based on size limits."""
    valid_problems = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n = int(row['n'])
                m = int(row['m'])
                if n <= MAX_VARIABLES and m <= MAX_CONSTRAINTS:
                    valid_problems.append(row['Name'])
            except (ValueError, KeyError) as e:
                print(f"Skipping invalid row: {row.get('Name', 'Unknown')} - {str(e)}")
    return valid_problems

# def finite_difference_gradient(func, x, epsilon):
#     """Calculate finite difference gradient."""
#     n = len(x)
#     grad = np.zeros(n)
#     for i in range(n):
#         x_plus = x.copy()
#         x_plus[i] += epsilon
#         f_plus = func(x_plus)
        
#         x_minus = x.copy()
#         x_minus[i] -= epsilon
#         f_minus = func(x_minus)
        
#         grad[i] = (f_plus - f_minus) / (2 * epsilon)
#     return grad

def setup_optimizer(problem):
    """Configure optimizer with constraints."""
    n = problem.n
    m = problem.m
    
    # Initialize optimizers
    opt = nlopt.opt(nlopt.AUGLAG, n)
    local_opt = nlopt.opt(nlopt.LD_SLSQP, n)
    
    # Configure local optimizer
    local_opt.set_xtol_rel(OPTIMIZER_CONFIG["xtol_rel"])
    local_opt.set_ftol_rel(OPTIMIZER_CONFIG["ftol_rel"])
    local_opt.set_maxeval(OPTIMIZER_CONFIG["maxeval"] // 2)
    opt.set_local_optimizer(local_opt)
    
    # Set variable bounds
    bl = problem.bl if problem.bl is not None else [-np.inf] * n
    bu = problem.bu if problem.bu is not None else [np.inf] * n
    opt.set_lower_bounds(bl)
    opt.set_upper_bounds(bu)
    
    # Objective function
    def objective(x, grad):
        if grad.size > 0:
            f, g = problem.obj(x, gradient=True)
            grad[:] = g
        else:
            f = problem.obj(x, gradient=False)
        return f
    opt.set_min_objective(objective)

    # Constraints function
    def eq_constraint(x, grad, i):
        ci = problem.cons(x, i, gradient=False)
        if grad.size > 0:
            ci, Ji = problem.cons(x, i, gradient=True)
            grad[:] = Ji
        else:
            ci = problem.cons(x, i, gradient=False)
        return ci

    def lower_constraint(x, grad, i, cli):
        ci = problem.cons(x, i, gradient=False)
        if grad.size > 0:
            ci, Ji = problem.cons(x, i, gradient=True)
            grad[:] = -Ji
        else:
            ci = problem.cons(x, i, gradient=False)
        return cli - ci

    def upper_constraint(x, grad, i, cui):
        ci = problem.cons(x, i, gradient=False)
        if grad.size > 0:
            ci, Ji = problem.cons(x, i, gradient=True)
            grad[:] = Ji
        else:
            ci = problem.cons(x, i, gradient=False)
        return ci - cui
    
    # Add constraints
    for i in range(m):
        if problem.is_eq_cons[i]:
            opt.add_equality_constraint(lambda x, grad: eq_constraint(x, grad, i), 1e-6)
        else:
            cl_i = problem.cl[i] if problem.cl is not None else -np.inf
            cu_i = problem.cu[i] if problem.cu is not None else np.inf
            # Lower constraint
            if cl_i > -np.inf:
                opt.add_inequality_constraint(lambda x, grad: lower_constraint(x, grad, i, cl_i), 1e-6)
            # Upper constraint
            if cu_i < np.inf:
                opt.add_inequality_constraint(lambda x, grad: upper_constraint(x, grad, i, cu_i), 1e-6)
    
    # Set optimizer parameters
    opt.set_xtol_rel(OPTIMIZER_CONFIG["xtol_rel"])
    opt.set_ftol_rel(OPTIMIZER_CONFIG["ftol_rel"])
    opt.set_maxeval(OPTIMIZER_CONFIG["maxeval"])
    opt.set_maxtime(OPTIMIZER_CONFIG["maxtime"])
    
    return opt

def evaluate_problem(problem_name):
    """Evaluate a single problem and return results."""
    result = {
        "problem": problem_name,
        "n": np.nan, "m": np.nan,
        "success": False,
        "objective": np.nan,
        "infeasibility": np.nan,
        "evaluations": np.nan,
        "time": np.nan,
        "exit_code": None,
        "message": None
    }
    
    try:
        # Load problem
        problem = pycutest.import_problem(problem_name)
        result["n"], result["m"] = problem.n, problem.m
        
        # Initialize optimizer
        opt = setup_optimizer(problem)
        x0 = problem.x0.copy()
        
        # Run optimization
        start_time = time.time()
        try:
            x_opt = opt.optimize(x0)
            exit_code = opt.last_optimize_result()
            min_value = opt.last_optimum_value()
            eval_count = opt.get_numevals()
            
            # Calculate constraint violation
            infeasibility = 0.0
            for i in range(problem.m):
                ci = problem.cons(x_opt, i, gradient=False)
                if problem.is_eq_cons[i]:
                    infeasibility += ci**2
                else:
                    if problem.cl is not None and (cl_i := problem.cl[i]) is not None:
                        infeasibility += max(cl_i - ci, 0)**2
                    if problem.cu is not None and (cu_i := problem.cu[i]) is not None:
                        infeasibility += max(ci - cu_i, 0)**2
            
            # Record results
            result.update({
                "success": exit_code in SUCCESS_CODES if infeasibility < 1e-2 else False,
                "objective": min_value,
                "infeasibility": np.sqrt(infeasibility) if infeasibility > 0 else 0.0,
                "evaluations": eval_count,
                "time": time.time() - start_time,
                "exit_code": exit_code,
                "message": SUCCESS_CODES.get(exit_code, "Optimization failed") if infeasibility < 1e-2 else "Infeasible solution"
            })
            
        except Exception as e:
            result["message"] = f"Optimization error: {str(e)}"
            
    except Exception as e:
        result["message"] = f"Problem loading error: {str(e)}"
    
    return result

if __name__ == "__main__":
    # Load filtered problem names from CSV
    problem_names = load_filtered_problems(CUTEST_METADATA)
    print(f"Found {len(problem_names)} problems within size limits.")
    
    # Create results file
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "problem", "n", "m", "success", 
            "objective", "infeasibility", 
            "evaluations", "time", 
            "exit_code", "message"
        ])
        writer.writeheader()
        
        # Process each problem
        for pname in problem_names:
            print(f"\n{'=' * 40}")
            print(f"Processing problem: {pname}")
            print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Evaluate problem
            result = evaluate_problem(pname)
            
            # Write results
            writer.writerow(result)
            f.flush()
            
            # Print progress
            print(f"Status: {result['message']}")
            if not np.isnan(result['n']):
                print(f"Variables: {int(result['n'])}, Constraints: {int(result['m'])}")
                obj_str = f"{result['objective']:.4e}" if not np.isnan(result['objective']) else "N/A"
                infeas_str = f"{result['infeasibility']:.2e}" if not np.isnan(result['infeasibility']) else "N/A"
                print(f"Objective: {obj_str}")
                print(f"Infeasibility: {infeas_str}")
                print(f"Time: {result['time']:.1f} seconds")