import pandas as pd
import matplotlib.pyplot as plt

# load the results
nlopt_df = pd.read_csv('../results/nlopt_pycutest_results.csv')
solnp_df = pd.read_csv('../results/solnp_results.csv')
nlopt_with_grad_df = pd.read_csv('../results/nlopt_pycutest_with_grad_results.csv')

# get problems solved by each solver

nlopt_solved = nlopt_df[nlopt_df['success'] == True]
solnp_solved = solnp_df[solnp_df['status'] == '1']
nlopt_with_grad_solved = nlopt_with_grad_df[nlopt_with_grad_df['success'] == True]

# get the number of problems solved by each solver

nlopt_solved_count = nlopt_solved['problem'].nunique()
solnp_solved_count = solnp_solved['problem'].nunique()
nlopt_with_grad_solved_count = nlopt_with_grad_solved['problem'].nunique()

print(f"Problems solved by NLOPT: {nlopt_solved_count}")
print(f"Problems solved by SOLNP+: {solnp_solved_count}")
print(f"Problems solved by NLOPT with gradient: {nlopt_with_grad_solved_count}")

common_nlopt = nlopt_solved.merge(nlopt_with_grad_solved, on='problem', how='inner')

common_nlopt_count = common_nlopt['problem'].nunique()

print(f"Common problems solved by NLOPT and NLOPT with gradient: {common_nlopt_count}")

# which problems are solved by both solvers

common_problems = nlopt_solved.merge(solnp_solved, on='problem', how='inner')

# get the number of common problems

common_problems_count = common_problems['problem'].nunique()

print(f"Common problems solved by both solvers: {common_problems_count}")

# problem can be solved by one solver but not the other

nlopt_only = nlopt_solved[~nlopt_solved['problem'].isin(common_problems['problem'])]
solnp_only = solnp_solved[~solnp_solved['problem'].isin(common_problems['problem'])]

nlopt_only_count = nlopt_only['problem'].nunique()
solnp_only_count = solnp_only['problem'].nunique()

print(f"Problems solved by NLOPT only: {nlopt_only_count}")
print(f"Problems solved by SOLNP+ only: {solnp_only_count}")

# average number of variables and constraints for problems solved by each solver

print("="*50)

nlopt_avg_vars = nlopt_solved['n'].mean()
nlopt_avg_cons = nlopt_solved['m'].mean()

print(f"Average number of variables for problems solved by NLOPT: {nlopt_avg_vars}")
print(f"Average number of constraints for problems solved by NLOPT: {nlopt_avg_cons}")

solnp_avg_vars = solnp_solved['n'].mean()
solnp_avg_cons = solnp_solved['m'].mean()

print(f"Average number of variables for problems solved by SOLNP+: {solnp_avg_vars}")
print(f"Average number of constraints for problems solved by SOLNP+: {solnp_avg_cons}")

# average number of variables and constraints for problems that can be solved by one solver but not the other

print("="*50)

nlopt_only_avg_vars = nlopt_only['n'].mean()
nlopt_only_avg_cons = nlopt_only['m'].mean()

print(f"Average number of variables for problems solved by NLOPT only: {nlopt_only_avg_vars}")
print(f"Average number of constraints for problems solved by NLOPT only: {nlopt_only_avg_cons}")

solnp_only_avg_vars = solnp_only['n'].mean()
solnp_only_avg_cons = solnp_only['m'].mean()

print(f"Average number of variables for problems solved by SOLNP+ only: {solnp_only_avg_vars}")
print(f"Average number of constraints for problems solved by SOLNP+ only: {solnp_only_avg_cons}")

# maximum number of variables and constraints for problems solved by each solver

print("="*50)

nlopt_max_vars = nlopt_solved['n'].max()
nlopt_max_cons = nlopt_solved['m'].max()

print(f"Maximum number of variables for problems solved by NLOPT: {nlopt_max_vars}")
print(f"Maximum number of constraints for problems solved by NLOPT: {nlopt_max_cons}")

solnp_max_vars = solnp_solved['n'].max()
solnp_max_cons = solnp_solved['m'].max()

print(f"Maximum number of variables for problems solved by SOLNP+: {solnp_max_vars}")
print(f"Maximum number of constraints for problems solved by SOLNP+: {solnp_max_cons}")

# the status of SOLNP+ for problems only solved by NLOPT

print("="*50)

nlopt_problem_names = nlopt_only['problem'].tolist()
solnp_nlopt_only = solnp_df[solnp_df['problem'].isin(nlopt_problem_names)]

solnp_nlopt_only_status = solnp_nlopt_only['status'].value_counts()

print(f"SOLNP+ status for problems solved by NLOPT only:")
print(solnp_nlopt_only_status)

# # print the problems only solved by NLOPT with SOLNP+ status 0

# print("="*50)

# solnp_nlopt_only_status_0 = solnp_nlopt_only[solnp_nlopt_only['status'] == '0']

# print("Problems solved by NLOPT only with SOLNP+ status 0:")
# print(solnp_nlopt_only_status_0['problem'].tolist())

# print the objective values for problems solved by NLOPT only with SOLNP+ status 0

# print("="*50)

# solnp_nlopt_only_status_0_obj = solnp_nlopt_only_status_0['obj']

# print("Objective values for problems solved by NLOPT only with SOLNP+ status 0:")
# print(solnp_nlopt_only_status_0_obj)

# print the infeasibility values for problems solved by NLOPT only with SOLNP+ status 0

# print("="*50)

# solnp_nlopt_only_status_0_infeas = solnp_nlopt_only_status_0['constraint']

# print("Infeasibility values for problems solved by NLOPT only with SOLNP+ status 0:")
# print(solnp_nlopt_only_status_0_infeas)

# print(f"Number of infeasible: {solnp_nlopt_only_status_0[solnp_nlopt_only_status_0['constraint'] > 1e-6].shape[0]}")

# # print the problems only solved by SOLNP+

# print("="*50)

# print("Problems solved by SOLNP+ only:")
# print(solnp_only['problem'].tolist())

# # print the common problems solved by both solvers

# print("="*50)

# print("Common problems solved by both solvers:")
# print(common_problems['problem'].tolist())

# plot the evaluations per iteration vs n for SOLNP+ on the solved problems

plt.figure(figsize=(10, 6))
plt.scatter(solnp_solved['n'], solnp_solved['count_cost'] / solnp_solved['iter'], color='blue')
plt.xlabel('Number of variables')
plt.ylabel('Evaluations per iteration')
plt.title('Evaluations per iteration vs n for SOLNP+ on solved problems')
plt.savefig('../results/solnp_evals_per_iter_vs_n.png')
plt.show()
