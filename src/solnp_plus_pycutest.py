import pycutest
from pysolnp import SOLNP
import numpy as np
import csv
import time
from datetime import datetime
import os
import multiprocessing as mp

# Configuration parameters
MAX_VARIABLES = 500       # Maximum number of variables
MAX_CONSTRAINTS = 1000    # Maximum number of constraints
RESULTS_FILE = "../results/solnp_results.csv"  # Results file name
CUTEST_METADATA = "../cutest.csv"      # CUTEst metadata file

# SOLNP+ configuration
SOLNP_CONFIG = {
    "tol": 1e-4,          # Convergence tolerance
    "tol_con": 1e-6,      # Infeasibility tolerance
    "maxfev": 1000,        # Maximum function evaluations
    "max_iter": 50        # Maximum iterations
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

def evaluate_problem(problem_name):
    """Evaluate a single problem using SOLNP+ and return results."""
    result = {
        "problem": problem_name,
        "n": np.nan, "m": np.nan,
        "status": None,  # Status of the algorithm
        "obj": np.nan,  # Objective function value
        "constraint": np.nan,  # Norm of constraints
        "iter": np.nan,  # Number of outer iterations
        "count_cost": np.nan,  # Number of function evaluations
        "total_time": np.nan,  # Total running time
        "message": None  # Additional message or error
    }
    
    try:
        # Load problem
        problem = pycutest.import_problem(problem_name)
        result["n"], result["m"] = problem.n, problem.m
        
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
        start_time = time.time()
        mysolnp = SOLNP(prob=prob, cost=cost, op=SOLNP_CONFIG)
        info = mysolnp.run()
        elapsed_time = time.time() - start_time
        
        # Populate results
        result.update({
            "status": info['status'],  # Algorithm status
            "obj": info['obj'],  # Objective function value
            "constraint": info.get('constraint', np.nan),  # Norm of constraints
            "iter": info['iter'],  # Number of outer iterations
            "count_cost": info.get('count_cost', np.nan),  # Function evaluations
            "total_time": elapsed_time,  # Total running time
            "message": info.get('message', '')  # Additional message
        })
        
    except Exception as e:
        result["message"] = f"Error: {str(e)}"
    
    return result

def run_solnp(pname):
    # Evaluate problem
    result = evaluate_problem(pname)
    with open(RESULTS_FILE, "a", newline="") as f:
        fieldnames = [
            "problem", "n", "m", "status", "obj", "constraint", "iter", "count_cost", "total_time", "message"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # Write results
        writer.writerow(result)
        f.flush()

    # Print progress
    print(f"Status: {result['status']}")
    if not np.isnan(result['n']):
        print(f"Variables: {int(result['n'])}, Constraints: {int(result['m'])}")
        obj_str = f"{result['obj']:.4e}" if not np.isnan(result['obj']) else "N/A"
        print(f"Objective: {obj_str}")
        print(f"Iterations: {result['iter']}")
        print(f"Total time: {result['total_time']:.1f} seconds")

if __name__ == "__main__":
    # Load filtered problem names from CSV
    problem_names = load_filtered_problems(CUTEST_METADATA)
    print(f"Found {len(problem_names)} problems within size limits.")
    
    # Create results file
    with open(RESULTS_FILE, "w", newline="") as f:
        fieldnames = [
            "problem", "n", "m", "status", "obj", "constraint", "iter", "count_cost", "total_time", "message"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
    # Process each problem
    for pname in problem_names:
        print(f"\n{'=' * 40}")
        print(f"Processing problem: {pname}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # create a new process to run the solver

        p = mp.Process(target=run_solnp, args=(pname,))
        p.start()
        p.join()
        if p.exitcode != 0:
            print(f"Process {pname} failed with exit code {p.exitcode}")
            result = {
                "problem": pname,
                "n": np.nan, "m": np.nan,
                "status": "Failed",
                "obj": np.nan,
                "constraint": np.nan,
                "iter": np.nan,
                "count_cost": np.nan,
                "total_time": np.nan,
                "message": "Process failed"
            }
            with open(RESULTS_FILE, "a", newline="") as f:
                fieldnames = [
                    "problem", "n", "m", "status", "obj", "constraint", "iter", "count_cost", "total_time", "message"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(result)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 40}")