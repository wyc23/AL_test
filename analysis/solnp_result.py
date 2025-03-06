# import re
# import matplotlib.pyplot as plt

# def parse_log_file(filename):
#     rho = [] # rho
#     pen_l1 = [] # l1 penalty
#     obj = [] # objective value
#     cons_nm2 = [] # norm of constraints
#     # iteration_pattern = re.compile(r'^SOLNP\+\-\-> Iteration (\d+):')
#     rho_pattern = re.compile(r'^\s*rho\s*=\s*([\d\.eE+-]+)')
#     pen_l1_pattern = re.compile(r'^\s*pen_l1\s*=\s*([\d\.eE+-]+)')
#     obj_pattern = re.compile(r'^\s*obj\s*=\s*([\d\.eE+-]+)')
#     cons_nm2_pattern = re.compile(r'^\s*cons_nm2\s*=\s*([\d\.eE+-]+)')
    
#     with open(filename, 'r') as f:
#         for line in f:
#             line = line.strip()
#             # 检查是否为迭代行
#             if rho_pattern.match(line):
#                 rho.append(float(rho_pattern.match(line).group(1)))
#                 continue
#             if pen_l1_pattern.match(line):
#                 pen_l1.append(float(pen_l1_pattern.match(line).group(1)))
#                 continue
#             if obj_pattern.match(line):
#                 obj.append(float(obj_pattern.match(line).group(1)))
#                 continue
#             if cons_nm2_pattern.match(line):
#                 cons_nm2.append(float(cons_nm2_pattern.match(line).group(1)))
#                 continue
#     return rho, pen_l1, obj, cons_nm2

# def plot_data_nocons(rho, pen_l1, obj):
#     plt.figure(figsize=(10, 8))

#     plt.subplot(3, 1, 1)
#     plt.plot(rho, 'b-')

#     plt.subplot(3, 1, 2)
#     plt.plot(pen_l1, 'g-')

#     plt.subplot(3, 1, 3)
#     plt.plot(obj, 'r-')

#     plt.show()

# def plot_data_cons(rho, pen_l1, obj, cons_nm2):
#     plt.figure(figsize=(10, 8))

#     plt.subplot(4, 1, 1)
#     plt.plot(rho, 'b-')

#     plt.subplot(4, 1, 2)
#     plt.plot(pen_l1, 'g-')

#     plt.subplot(4, 1, 3)
#     plt.plot(obj, 'r-')

#     plt.subplot(4, 1, 4)
#     plt.plot(cons_nm2, 'y-')

#     plt.show()

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) < 2:
#         print("Usage: python plot_solnp_log.py <filename>")
#         sys.exit(1)
#     rho, pen_l1, obj, cons_nm2 = parse_log_file(sys.argv[1])
#     print(f"len(rho) = {len(rho)}")
#     print(f"len(pen_l1) = {len(pen_l1)}")
#     print(f"len(obj) = {len(obj)}")
#     print(f"len(cons_nm2) = {len(cons_nm2)}")
#     if len(cons_nm2) == 0:
#         plot_data_nocons(rho, pen_l1, obj)
#     else:
#         plot_data_cons(rho, pen_l1, obj, cons_nm2)



# import re
# import matplotlib.pyplot as plt

# def parse_log_file(filename):
#     rho = [] # rho
#     pen_l1 = [] # l1 penalty
#     obj = [] # objective value
#     cons_nm2 = [] # norm of constraints
#     # iteration_pattern = re.compile(r'^SOLNP\+\-\-> Iteration (\d+):')
#     rho_pattern = re.compile(r'^\s*rho\s*=\s*([\d\.eE+-]+)')
#     pen_l1_pattern = re.compile(r'^\s*pen_l1\s*=\s*([\d\.eE+-]+)')
#     obj_pattern = re.compile(r'^\s*obj\s*=\s*([\d\.eE+-]+)')
#     cons_nm2_pattern = re.compile(r'^\s*cons_nm2\s*=\s*([\d\.eE+-]+)')
    
#     with open(filename, 'r') as f:
#         for line in f:
#             line = line.strip()
#             # 检查是否为迭代行
#             if rho_pattern.match(line):
#                 rho.append(float(rho_pattern.match(line).group(1)))
#                 continue
#             if pen_l1_pattern.match(line):
#                 pen_l1.append(float(pen_l1_pattern.match(line).group(1)))
#                 continue
#             if obj_pattern.match(line):
#                 obj.append(float(obj_pattern.match(line).group(1)))
#                 continue
#             if cons_nm2_pattern.match(line):
#                 cons_nm2.append(float(cons_nm2_pattern.match(line).group(1)))
#                 continue
#     return rho, pen_l1, obj, cons_nm2

# def plot_data_nocons(rho, pen_l1, obj):
#     plt.figure(figsize=(10, 8))

#     plt.subplot(3, 1, 1)
#     plt.plot(rho, 'b-', label='rho')
#     plt.title('Rho over Iterations')
#     plt.xlabel('Iteration')
#     plt.ylabel('Rho')
#     plt.grid(True)
#     plt.legend()

#     plt.subplot(3, 1, 2)
#     plt.plot(pen_l1, 'g-', label='L1 Penalty')
#     plt.title('L1 Penalty over Iterations')
#     plt.xlabel('Iteration')
#     plt.ylabel('L1 Penalty')
#     plt.grid(True)
#     plt.legend()

#     plt.subplot(3, 1, 3)
#     plt.plot(obj, 'r-', label='Objective Value')
#     plt.title('Objective Value over Iterations')
#     plt.xlabel('Iteration')
#     plt.ylabel('Objective Value')
#     plt.grid(True)
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

# def plot_data_cons(rho, pen_l1, obj, cons_nm2):
#     plt.figure(figsize=(10, 8))

#     plt.subplot(4, 1, 1)
#     plt.plot(rho, 'b-', label='rho')
#     plt.title('Rho over Iterations')
#     plt.xlabel('Iteration')
#     plt.ylabel('Rho')
#     plt.grid(True)
#     plt.legend()

#     plt.subplot(4, 1, 2)
#     plt.plot(pen_l1, 'g-', label='L1 Penalty')
#     plt.title('L1 Penalty over Iterations')
#     plt.xlabel('Iteration')
#     plt.ylabel('L1 Penalty')
#     plt.grid(True)
#     plt.legend()

#     plt.subplot(4, 1, 3)
#     plt.plot(obj, 'r-', label='Objective Value')
#     plt.title('Objective Value over Iterations')
#     plt.xlabel('Iteration')
#     plt.ylabel('Objective Value')
#     plt.grid(True)
#     plt.legend()

#     plt.subplot(4, 1, 4)
#     plt.plot(cons_nm2, 'y-', label='Constraint Norm')
#     plt.title('Constraint Norm over Iterations')
#     plt.xlabel('Iteration')
#     plt.ylabel('Constraint Norm')
#     plt.grid(True)
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) < 2:
#         print("Usage: python plot_solnp_log.py <filename>")
#         sys.exit(1)
#     rho, pen_l1, obj, cons_nm2 = parse_log_file(sys.argv[1])
#     print(f"len(rho) = {len(rho)}")
#     print(f"len(pen_l1) = {len(pen_l1)}")
#     print(f"len(obj) = {len(obj)}")
#     print(f"len(cons_nm2) = {len(cons_nm2)}")
#     if len(cons_nm2) == 0:
#         plot_data_nocons(rho, pen_l1, obj)
#     else:
#         plot_data_cons(rho, pen_l1, obj, cons_nm2)


import os
import re
import matplotlib.pyplot as plt

def parse_log_file(filename):
    rho = []  # rho
    pen_l1 = []  # l1 penalty
    obj = []  # objective value
    cons_nm2 = []  # norm of constraints
    rho_pattern = re.compile(r'^\s*rho\s*=\s*([\d\.eE+-]+)')
    pen_l1_pattern = re.compile(r'^\s*pen_l1\s*=\s*([\d\.eE+-]+)')
    obj_pattern = re.compile(r'^\s*obj\s*=\s*([\d\.eE+-]+)')
    cons_nm2_pattern = re.compile(r'^\s*cons_nm2\s*=\s*([\d\.eE+-]+)')
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if rho_pattern.match(line):
                rho.append(float(rho_pattern.match(line).group(1)))
            if pen_l1_pattern.match(line):
                pen_l1.append(float(pen_l1_pattern.match(line).group(1)))
            if obj_pattern.match(line):
                try:
                    obj.append(float(obj_pattern.match(line).group(1)))
                except ValueError:
                    try:
                        obj.append(obj[-1])
                    except IndexError:
                        obj.append(-100000)
            if cons_nm2_pattern.match(line):
                try:
                    cons_nm2.append(float(cons_nm2_pattern.match(line).group(1)))
                except ValueError:
                    try:
                        cons_nm2.append(cons_nm2[-1])
                    except IndexError:
                        cons_nm2.append(-1.0)
    return rho, pen_l1, obj, cons_nm2

def plot_data_nocons(rho, pen_l1, obj, filename):
    plt.figure(figsize=(12, 8))

    # 2x2 布局，三张图
    plt.subplot(2, 2, 1)
    plt.plot(rho, 'b-', label='Rho')
    plt.title('Rho over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Rho')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(pen_l1, 'g-', label='L1 Penalty')
    plt.title('L1 Penalty over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('L1 Penalty')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(obj, 'r-', label='Objective Value')
    plt.title('Objective Value over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.grid(True)
    plt.legend()

    # 隐藏第四个位置
    plt.subplot(2, 2, 4)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.splitext(filename)[0] + '.png')

def plot_data_cons(rho, pen_l1, obj, cons_nm2, filename):
    plt.figure(figsize=(12, 8))

    # 2x2 布局，四张图
    plt.subplot(2, 2, 1)
    plt.plot(rho, 'b-', label='Rho')
    plt.title('Rho over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Rho')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(pen_l1, 'g-', label='L1 Penalty')
    plt.title('L1 Penalty over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('L1 Penalty')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(obj, 'r-', label='Objective Value')
    plt.title('Objective Value over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(cons_nm2, 'y-', label='Constraint Norm')
    plt.title('Constraint Norm over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Constraint Norm')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.splitext(filename)[0] + '.png')

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plot_solnp_log.py <dirname> ")
        sys.exit(1)
    dirname = sys.argv[1]
    for filename in os.listdir(dirname):
        filename = os.path.join(dirname, filename)
        print("Processing", filename)
        rho, pen_l1, obj, cons_nm2 = parse_log_file(filename)
        print(f"len(rho) = {len(rho)}")
        print(f"len(pen_l1) = {len(pen_l1)}")
        print(f"len(obj) = {len(obj)}")
        print(f"len(cons_nm2) = {len(cons_nm2)}")
        if len(cons_nm2) == 0:
            plot_data_nocons(rho, pen_l1, obj, filename)
        else:
            plot_data_cons(rho, pen_l1, obj, cons_nm2, filename)