# sep_comb.py
def separate_comb(lp_sol, customers):
    """
    实现Comb不等式的分离算法，使用贪婪策略构造handle和tooth
    
    参数:
        lp_sol: LP解，字典，键为边(i,j)，值为对应的解值x_ij
        customers: 客户集合
    
    返回:
        violations: 列表，包含所有违反的不等式信息
    """
    violations = []
    violation_threshold = 1.0  # 违反阈值
    
    # 使用贪婪策略构造handle
    # 这里简化为选择一个随机的客户子集作为handle
    import random
    customers_set = set(customers)  # 将customers列表转换为集合
    H = set(random.sample(list(customers), len(customers) // 2))
    
    # 计算handle H的外部出弧总和
    H_out_sum = sum(lp_sol.get((i, j), 0) for i in H for j in lp_sol if j not in H)
    
    # 使用贪婪策略构造一个tooth
    # 这里简化为选择handle之外的一个随机客户子集作为tooth
    remaining = customers_set - H  # 使用转换后的集合进行运算
    T = set(random.sample(list(remaining), min(2, len(remaining))))
    
    # 计算tooth T的外部出弧总和
    T_out_sum = sum(lp_sol.get((i, j), 0) for i in T for j in lp_sol if j not in T)
    
    comb_value = H_out_sum + T_out_sum
    # 如果comb_value小于violation_threshold，则认为不等式被违反
    if comb_value < violation_threshold:
        violation = {
            'handle': H,
            'tooth': T,
            'comb_value': comb_value,
            'inequality': f"x(δ⁺(H)) + x(δ⁺(T)) >= {violation_threshold}"
        }
        violations.append(violation)
    
    return violations

if __name__ == '__main__':
    # 测试分离算法
    from instance import generate_instance
    from lp_solution import generate_lp_solution
    cm, D, V = generate_instance()
    lp = generate_lp_solution(cm, D, V)
    violations = separate_comb(lp, V)
    print("Detected comb inequality violations:")
    for v in violations:
        print(v)