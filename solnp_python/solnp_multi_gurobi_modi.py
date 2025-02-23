import pycutest
import numpy as np 
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import approx_fprime, minimize_scalar
import os
import pandas as pd  # 用于保存最终的表格

# 定义一个函数预处理变量上下界：超过 threshold 的边界视为无穷
def process_bounds(l_x, u_x, threshold=1e10):
    effective_l = np.where(np.abs(l_x) > threshold, -np.inf, l_x)
    effective_u = np.where(np.abs(u_x) > threshold, np.inf, u_x)
    return effective_l, effective_u

# 方法2：在 infeas 函数内部忽略过大边界（阈值设为1e10）
def infeas(x, eq_constraints, l_x, u_x, threshold=1e10):
    g = eq_constraints(x)
    infeas_eq = np.sum(g**2)
    # 对于上下界，超过阈值视为无穷
    effective_u = np.where(np.abs(u_x) > threshold, np.inf, u_x)
    effective_l = np.where(np.abs(l_x) > threshold, -np.inf, l_x)
    infeas_bounds = np.sum(np.maximum(0, x - effective_u)**2) + np.sum(np.maximum(0, effective_l - x)**2)
    return np.sqrt(infeas_eq + infeas_bounds)

# 定义问题集
problem_names = [  
                 'HS99'
    #  'HS10','HS11','HS12','HS13','HS14','HS15','HS16','HS17','HS18','HS19',
    #  'HS20','HS21','HS22','HS23','HS24','HS25','HS26','HS27','HS28','HS29',  
    #  'HS30','HS31','HS32','HS33','HS34','HS35','HS36','HS37','HS38','HS39',  
    #  'HS40','HS41','HS42','HS43','HS44','HS45','HS46','HS47','HS48','HS49',  
    #  'HS50','HS51','HS52','HS53','HS54','HS55','HS56','HS57', 'HS59'    ,
    #  'HS60','HS61','HS62','HS63','HS64','HS65','HS66','HS67','HS68','HS69',  
    #  'HS70','HS71','HS72','HS73','HS74','HS75','HS76','HS77','HS78','HS79',  
    #  'HS80','HS81','HS83','HS84','HS85','HS86','HS87','HS88','HS89',         
    #  'HS90','HS91','HS92','HS93','HS95','HS96','HS97','HS98','HS99',        
    #  'HS100','HS101','HS102','HS103','HS104','HS105','HS109',              
    #  'HS110','HS114'                                
]

# 创建保存结果的文件夹
output_folder = 'results_gurobi_modi'
os.makedirs(output_folder, exist_ok=True)

final_results = []
skipped_problems = []

for problem_name in problem_names:
    # 从 PyCUTEst 读取问题
    problem = pycutest.import_problem(problem_name)

    # 定义目标函数
    def objective_function(x):
        return problem.obj(x)

    # 定义等式约束
    def equality_constraints(x):
        if problem.m > 0:
            eq_constraints = []
            for i in range(problem.m):
                if problem.is_eq_cons[i]:
                    ci = problem.cons(x, index=i)
                    eq_constraints.append(ci)
            return np.array(eq_constraints)
        return np.array([])

    # 定义不等式约束，返回形式为 g(x) >= 0
    # 对于默认边界超过阈值的情况，将其视为无穷
    def inequality_constraints(x, bound_threshold=1e10):
        if problem.m > 0:
            ineq_constraints = []
            for i in range(problem.m):
                if not problem.is_eq_cons[i]:
                    ci = problem.cons(x, index=i)
                    cl = problem.cl[i] if abs(problem.cl[i]) < bound_threshold else -np.inf
                    cu = problem.cu[i] if abs(problem.cu[i]) < bound_threshold else np.inf
                    if cl > -np.inf:
                        ineq_constraints.append(ci - cl)
                    if cu < np.inf:
                        ineq_constraints.append(cu - ci)
            return np.array(ineq_constraints)
        return np.array([])

    # 获取变量上下界，并进行预处理
    l_x_orig = problem.bl if problem.bl is not None else -np.inf * np.ones(problem.n)
    u_x_orig = problem.bu if problem.bu is not None else np.inf * np.ones(problem.n)
    l_x, u_x = process_bounds(l_x_orig, u_x_orig, threshold=1e10)

    # 设定初始解
    p0 = problem.x0

    def get_problem_data():
        return objective_function, equality_constraints, inequality_constraints, l_x, u_x, p0

    # 增广拉格朗日函数（只考虑等式约束部分）
    def augmented_lagrangian(x, y, rho, obj_func, eq_constraints):
        lagr = obj_func(x[:problem.n])
        g = eq_constraints(x)
        lagr -= np.dot(y, g)
        lagr += (rho / 2) * np.linalg.norm(g)**2
        return lagr

    # 计算约束的雅可比矩阵
    def jacobian_eq_constraints(x, eq_constraints):
        epsilon = np.sqrt(np.finfo(float).eps)
        m1 = len(eq_constraints(x))
        n = len(x)
        jacobian = np.zeros((m1, n))
        for i in range(m1):
            jacobian[i, :] = approx_fprime(x, lambda x: eq_constraints(x)[i], epsilon)
        return jacobian

    # 找到一个内部（或近乎可行）的解
    def find_feasible_solution(p_k, eq_constraints, l_x, u_x):
        J = jacobian_eq_constraints(p_k, eq_constraints)
        gk = eq_constraints(p_k)
        if np.all(l_x < -1e+5) and np.all(u_x > 1e+5):
            p_f_k = p_k - np.linalg.pinv(J) @ gk
        else:
            model = gp.Model("feasible_solution")
            model.Params.TimeLimit = 90
            model.Params.FeasibilityTol = 1e-9
            model.Params.OutputFlag = 0

            n_vars = len(p_k)
            x = [model.addVar(lb=l_x[i], ub=u_x[i]) for i in range(n_vars)]
            tau = model.addVar(lb=0, ub=GRB.INFINITY)
            model.setObjective(tau, GRB.MINIMIZE)

            for i in range(len(gk)):
                model.addConstr(gp.quicksum(J[i, j] * (x[j] - p_k[j]) for j in range(n_vars)) - gk[i] * tau == -gk[i])
            model.optimize()

            if model.Status == GRB.OPTIMAL:
                p_f_k = np.array([x[i].X for i in range(n_vars)])
            else:
                p_f_k = p_k.copy()
        return p_f_k

    # 使用 Gurobi 求解 QP 子问题
    def solve_qp_subproblem_with_gurobi(xi_k, H, gradient_L, J, p_f_k, p_k, l_x, u_x, tol):
        n_vars = len(xi_k)
        model = gp.Model("qp_subproblem")
        model.Params.TimeLimit = 60
        model.Params.FeasibilityTol = tol
        model.Params.OutputFlag = 0

        x = [model.addVar(lb=l_x[i], ub=u_x[i]) for i in range(n_vars)]
        quad_expr = gp.QuadExpr()
        for i in range(n_vars):
            for j in range(n_vars):
                quad_expr += 0.5 * H[i, j] * (x[i] - xi_k[i]) * (x[j] - xi_k[j])
        lin_expr = gp.quicksum(gradient_L[i] * (x[i] - xi_k[i]) for i in range(n_vars))
        model.setObjective(quad_expr + lin_expr, GRB.MINIMIZE)

        constraint_rhs = J.dot(p_f_k - p_k)
        for i in range(len(constraint_rhs)):
            model.addConstr(gp.quicksum(J[i, j] * (x[j] - p_k[j]) for j in range(n_vars)) == constraint_rhs[i])
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            x_opt = np.array([x[i].X for i in range(n_vars)])
            return x_opt, True
        else:
            return xi_k, False

    def line_search(xi_k_old, xi_k, augmented_lagrangian, y, rho, obj_func, eq_constraints):
        def F(alpha):
            return augmented_lagrangian(alpha * xi_k_old + (1 - alpha) * xi_k, y, rho, obj_func, eq_constraints)
        result = minimize_scalar(F, bounds=(0, 1), method='bounded')
        alpha_star = result.x
        return alpha_star * xi_k_old + (1 - alpha_star) * xi_k

    def inner_iteration(p_f_k, xi_k, p_k, yk, rho, obj_func, eq_constraints, l_x, u_x, H, tol):
        n = len(p_f_k)
        m1 = len(eq_constraints(p_f_k))
        
        J = jacobian_eq_constraints(p_k, eq_constraints)
        gk = eq_constraints(p_k)
        gradient_L = approx_fprime(xi_k, lambda x: augmented_lagrangian(x, yk, rho, obj_func, eq_constraints), np.sqrt(np.finfo(float).eps))
        xi_k, success = solve_qp_subproblem_with_gurobi(xi_k, H, gradient_L, J, p_f_k, p_k, l_x, u_x, tol)
        if success:
            lambda_k = -np.linalg.pinv(J @ J.T) @ (J @ (xi_k - p_k) + gk)
            return xi_k, lambda_k, success
        else:
            return xi_k, np.zeros(m1), success

    def project_to_box(x, l_x, u_x):
        return np.minimum(np.maximum(x, l_x), u_x)

    def solnp_solver(obj_func, eq_constraints, ineq_constraints, l_x, u_x, p0, max_outer_iter=100, max_inner_iter=10, tol=1e-6):
        n = len(p0)
        m1 = len(eq_constraints(p0))
        m2 = len(ineq_constraints(p0))
        p0 = np.hstack((p0, np.zeros(m2)))
        l_x = np.hstack((l_x, np.zeros(m2)))
        u_x = np.hstack((u_x, np.inf * np.ones(m2)))
        print("n=", n)
        print("m1=", m1)
        print("m2=", m2)
        
        rho = 1.0
        H = np.eye(n + m2)
        y = np.zeros(m1 + m2)
        p_k = p0.copy()

        def extended_eq_constraints(x):
            return np.hstack((eq_constraints(x[:n]), ineq_constraints(x[:n]) - x[n:]))

        objective_values = []
        infeasibilities = []
        solutions = []
        rhos = []
        relative_diff_values = []
        box_project_norms = []

        # 保存初始信息
        init_obj = objective_function(p_k[:n])
        init_infeas = infeas(p_k, extended_eq_constraints, l_x, u_x)
        init_point = p_k[:n].copy()

        objective_values.append(init_obj)
        infeasibilities.append(init_infeas)
        rhos.append(rho)
        relative_diff_values.append(0)
        box_project_norms.append(0)

        c_z = 1.2
        c_ir = 10.0
        c_rr = 5.0
        r_ir = 5.0
        r_rr = 0.2
        epsilon_s = 1e-4
        epsilon_a = 1e-2

        print(f"Initial solution: {init_point}")
        print(f"Initial objective value: {init_obj}")
        print(f"Initial infeasibility: {init_infeas}")

        for k in range(max_outer_iter):
            v_k = infeas(p_k, extended_eq_constraints, l_x, u_x)
            if v_k <= c_z * tol:
                rho = 0.0

            p_f_k = find_feasible_solution(p_k, extended_eq_constraints, l_x, u_x)
            xi_k = p_f_k.copy()

            for i in range(max_inner_iter):
                xi_k_old = xi_k.copy()
                xi_k, lagrange_multiplier, success = inner_iteration(
                    p_f_k, xi_k, p_k, y, rho, obj_func, extended_eq_constraints, l_x, u_x, H, tol)
                if not success:
                    print(f"Inner iteration {i+1} failed.")
                    break
                sk = xi_k - xi_k_old
                t_k = approx_fprime(xi_k, lambda x: augmented_lagrangian(x, y, rho, obj_func, extended_eq_constraints), np.sqrt(np.finfo(float).eps)) - \
                      approx_fprime(xi_k_old, lambda x: augmented_lagrangian(x, y, rho, obj_func, extended_eq_constraints), np.sqrt(np.finfo(float).eps))
                if np.dot(sk, t_k) > 0:
                    H = H + np.outer(t_k, t_k) / np.dot(t_k, sk) - np.dot(H, np.outer(sk, sk)).dot(H) / np.dot(sk, H.dot(sk))
                y = lagrange_multiplier
                xi_k = line_search(xi_k_old, xi_k, augmented_lagrangian, y, rho, obj_func, extended_eq_constraints)
                print(f"Inner iteration {i+1}, xi_k: {xi_k[:n]}, infeasibility: {infeas(xi_k, extended_eq_constraints, l_x, u_x)}")
                print(f"Lagrange multipliers: {lagrange_multiplier}")
                print(f"Hessian matrix H:\n{H}")
                if np.linalg.norm(sk) < tol:
                    print(f"Inner iteration {i+1} converged.")
                    break

            gk = extended_eq_constraints(xi_k)
            y = y - rho * gk
            v_k_new = infeas(xi_k, extended_eq_constraints, l_x, u_x)
            if v_k_new >= c_ir * v_k:
                rho = r_ir * rho
            elif v_k_new <= c_rr * v_k:
                rho = r_rr * rho

            objective_values.append(objective_function(xi_k[:n]))
            infeasibilities.append(v_k_new)
            solutions.append(xi_k[:n].copy())
            rhos.append(rho)
            relative_diff_values.append(abs(objective_function(xi_k[:n]) - objective_function(p_k[:n])) / max(1, abs(objective_function(p_k[:n]))))
            box_project_norms.append(np.linalg.norm(project_to_box(xi_k - approx_fprime(xi_k, lambda x: augmented_lagrangian(x, y, rho, obj_func, extended_eq_constraints), np.sqrt(np.finfo(float).eps)), l_x, u_x) - xi_k))
            if (relative_diff_values[-1] <= epsilon_s and box_project_norms[-1] > epsilon_a):
                print(f"Restarting at outer iteration {k+1}")
                H = np.diag(np.diag(H))
                continue

            if objective_function(xi_k[:n]) > objective_function(p_k[:n]) and v_k_new > v_k:
                print(f"Restarting due to increase in objective and infeasibility at outer iteration {k+1}")
                y = np.zeros(m1 + m2)
                H = np.diag(np.diag(H))
                continue

            print(f"Outer iteration {k+1}, optimal solution: {xi_k[:n]}, infeasibility: {infeas(xi_k, extended_eq_constraints, l_x, u_x)}")
            if v_k_new < tol and relative_diff_values[-1] < epsilon_s:
                print(f"Outer iteration {k+1} converged.")
                break
            p_k = xi_k.copy()

        return p_k[:n], objective_values, infeasibilities, rhos, relative_diff_values, box_project_norms

    if __name__ == "__main__":
        try:
            solution, objective_values, infeasibilities, rhos, relative_diff_values, box_project_norms = solnp_solver(objective_function, equality_constraints, inequality_constraints, l_x, u_x, p0)
            final_obj_value = objective_values[-1]
            final_infeas_value = infeasibilities[-1]
            original_infeas_value = infeasibilities[0]
            final_results.append({"Problem": problem_name, 
                                   "Final Objective Value": final_obj_value, 
                                   "Initial Objective Value": objective_values[0],
                                   "Original Infeasibility": original_infeas_value, 
                                   "Final Infeasibility": final_infeas_value})
            print(f"Initial point: {p0}")
            print(f"Final optimal solution for {problem_name}:", solution)
            print(f"Initial objective value: {objective_values[0]}")
            print(f"Final objective value for {problem_name}:", final_obj_value)
            print(f"Initial infeasibility: {original_infeas_value}")
            print(f"Final infeasibility: {final_infeas_value}")

            plt.figure(figsize=(12, 12))
            plt.subplot(2, 2, 1)
            plt.plot(range(len(objective_values)), objective_values, label='Objective Function Value')
            plt.xlabel('Outer Iteration')
            plt.ylabel('Objective Function Value')
            plt.title(f'Objective Function Value over Iterations for {problem_name}')
            plt.legend()
            plt.subplot(2, 2, 2)
            plt.plot(range(len(infeasibilities)), infeasibilities, label='Infeasibility')
            plt.xlabel('Outer Iteration')
            plt.ylabel('Infeasibility')
            plt.title(f'Infeasibility over Iterations for {problem_name}')
            plt.legend()
            plt.subplot(2, 2, 3)
            plt.plot(range(len(relative_diff_values)), relative_diff_values, label='Relative Difference')
            plt.xlabel('Outer Iteration')
            plt.ylabel('Relative Difference')
            plt.title(f'Relative Difference over Iterations for {problem_name}')
            plt.legend()
            plt.subplot(2, 2, 4)
            plt.plot(range(len(box_project_norms)), box_project_norms, label='Box Projection Norm')
            plt.xlabel('Outer Iteration')
            plt.ylabel('Box Projection Norm')
            plt.title(f'Box Projection Norm over Iterations for {problem_name}')
            plt.legend()
            plt.suptitle(f'Optimization Results for {problem_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f'{problem_name}_results_exact_hess.png'))
            plt.close()
        except gp.GurobiError as e:
            print(f"Skipping problem {problem_name} due to GurobiError: {e}")
            skipped_problems.append(problem_name)
            final_results.append({"Problem": problem_name, "Final Objective Value": "Skipped"})
        except ValueError as e:
            print(f"Skipping problem {problem_name} due to error: {e}")
            skipped_problems.append(problem_name)
            final_results.append({"Problem": problem_name, "Final Objective Value": "Skipped"})

final_results_df = pd.DataFrame(final_results)
final_results_df.to_csv(os.path.join(output_folder, 'results.csv'), index=False)

if skipped_problems:
    print("Skipped problems due to errors:")
    for problem in skipped_problems:
        print(problem)

with open(os.path.join(output_folder, 'skipped_problems.txt'), 'w') as f:
    for problem in skipped_problems:
        f.write(f"{problem}\n")
