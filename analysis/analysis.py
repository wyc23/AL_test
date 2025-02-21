import pandas as pd

# load the results
nlopt_df = pd.read_csv('../results/nlopt_pycutest_results.csv')
solnp_df = pd.read_csv('../results/solnp_results.csv')

# get problems solved by each solver

nlopt_solved = nlopt_df[nlopt_df['success'] == True]
solnp_solved = solnp_df[solnp_df['status'] == '1']

# get the number of problems solved by each solver

nlopt_solved_count = nlopt_solved['problem'].nunique()
solnp_solved_count = solnp_solved['problem'].nunique()

print(f"Problems solved by NLOPT: {nlopt_solved_count}")
print(f"Problems solved by SOLNP+: {solnp_solved_count}")

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

# print the problems only solved by NLOPT

print("="*50)

print("Problems solved by NLOPT only:")
print(nlopt_only['problem'].tolist())


