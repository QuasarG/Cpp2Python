# lp_solution.py
import numpy as np

def generate_lp_solution_amdTSP(cost_matrix, depots, customers, max_routes_per_depot):
    """
    生成一个模拟的LP松弛解，用于测试A-MDTSP的分离算法
    
    参数:
        cost_matrix: 成本矩阵
        depots: 仓库集合
        customers: 客户集合
        max_routes_per_depot: 每个仓库最多的路径数（A-MDTSP中为1）
    
    返回:
        lp_sol: 字典，键为边(i,j)，值为对应的LP解值x_ij
    """
    np.random.seed(42)
    lp_sol = {}
    
    # 为每个仓库随机选择一个客户连接
    for d in depots:
        # 只允许每个仓库连接一个客户
        connected_customer = np.random.choice(customers)
        lp_sol[(d, connected_customer)] = 1.0
    
    # 生成客户之间的连接（LP松弛解）
    for i in customers:
        for j in customers:
            if i != j:
                lp_sol[(i, j)] = np.random.uniform(0.3, 0.7)  # 赋值随机LP解
    
    return lp_sol

if __name__ == '__main__':
    from instance import generate_instance_amdTSP
    cm, D, V, max_routes = generate_instance_amdTSP()
    lp = generate_lp_solution_amdTSP(cm, D, V, max_routes)
    print("A-MDTSP LP solution sample:")
    for key in sorted(list(lp.keys())[:10]):
        print(f"{key}: {lp[key]:.2f}")