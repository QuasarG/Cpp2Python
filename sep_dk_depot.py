# sep_dk_depot.py

def separate_depot_fixing_amdTSP(lp_sol, depots, customers, max_seq_length=3):
    """
    实现D+k/D-k仓库固定约束的分离算法
    
    参数:
        lp_sol: LP解，字典，键为边(i,j)，值为对应的解值x_ij
        depots: 仓库集合
        customers: 客户集合
        max_seq_length: 考虑的最大客户序列长度，默认为3
    
    返回:
        violations: 列表，包含所有违反的不等式信息
    """
    violations = []
    
    # 基本情况：检查客户对 (i, j)
    for i in customers:
        for j in customers:
            if i == j:
                continue
            # 计算仓库连接情况
            depot_sum = sum(lp_sol.get((i, s), 0) for s in depots) + sum(lp_sol.get((s, j), 0) for s in depots)
            if depot_sum > 1.0:  # A-MDTSP 限制，每个仓库最多一个路径
                violation = {
                    'sequence': [i, j],
                    'depot_sum': depot_sum,
                    'inequality': f"sum_[s in depots] [x({i},{s}) + x({s},{j})] <= 1"
                }
                violations.append(violation)
    
    # 使用DFS动态扩展客户序列，而不是枚举所有可能的排列
    def dfs_extend_sequence(current_seq, remaining_customers, depth):
        # 当序列长度达到目标长度时，检查约束
        if len(current_seq) >= 3 and len(current_seq) <= max_seq_length:
            # 使用贪心策略选择最优的仓库子集O
            check_depot_fixing_constraints(current_seq)
        
        # 如果已达到最大深度或没有剩余客户，则返回
        if len(current_seq) >= max_seq_length or not remaining_customers or depth >= max_seq_length:
            return
        
        # 尝试添加每个剩余客户到序列中
        for customer in remaining_customers:
            # 添加客户到序列
            current_seq.append(customer)
            # 递归扩展序列
            dfs_extend_sequence(current_seq, [c for c in remaining_customers if c != customer], depth + 1)
            # 回溯
            current_seq.pop()
    
    # 使用贪心策略选择最优的仓库子集O，而不是枚举所有可能的子集
    def check_depot_fixing_constraints(seq):
        # 贪心选择最优的仓库子集O
        O = set()
        D_minus_O = set(depots)
        
        # 对每个仓库，决定将其分配到O或D\O以最大化违反值
        for s in depots:
            # 计算将s分配到O的贡献
            contribution_to_O = lp_sol.get((seq[0], s), 0)
            # 计算将s分配到D\O的贡献
            contribution_to_D_minus_O = lp_sol.get((s, seq[-1]), 0)
            
            # 贪心选择：将s分配到贡献更大的集合
            if contribution_to_O > contribution_to_D_minus_O:
                O.add(s)
                D_minus_O.remove(s)
        
        # 如果O为空或等于全部仓库，则跳过（需要非平凡的划分）
        if not O or not D_minus_O:
            return
        
        # 计算D+k约束左端值
        k = len(seq)
        
        # 公式(30): sum_{s∈O} x_{i1,s} + sum_{s∈D\O} x_{s,ik} + sum_{h=2}^{k-1} x_{ih,i_{h-1}} + 2*sum_{h=2}^{k-1} x_{i1,ih}
        
        # 第一项：i1到O中仓库的边
        lhs_plus = sum(lp_sol.get((seq[0], s), 0) for s in O)
        
        # 第二项：D\O中仓库到ik的边
        lhs_plus += sum(lp_sol.get((s, seq[-1]), 0) for s in D_minus_O)
        
        # 第三项：中间客户之间的边
        for h in range(1, k-1):
            lhs_plus += lp_sol.get((seq[h], seq[h-1]), 0)
        
        # 第四项：i1到中间客户的边
        for h in range(2, k):
            lhs_plus += 2 * lp_sol.get((seq[0], seq[h]), 0)
        
        # 检查是否违反D+k约束
        if lhs_plus > k:
            violation = {
                'sequence': list(seq),
                'depots_O': list(O),
                'depots_D_minus_O': list(D_minus_O),
                'lhs_value': lhs_plus,
                'inequality': f"D+{k}: sum(...) <= {k}"
            }
            violations.append(violation)
        
        # 计算D-k约束左端值
        # 公式(31): sum_{s∈O} x_{s,i1} + sum_{s∈D\O} x_{ik,s} + sum_{h=2}^{k-1} x_{i_{h-1},ih} + 2*sum_{h=2}^{k-1} x_{ih,i1}
        
        # 第一项：O中仓库到i1的边
        lhs_minus = sum(lp_sol.get((s, seq[0]), 0) for s in O)
        
        # 第二项：ik到D\O中仓库的边
        lhs_minus += sum(lp_sol.get((seq[-1], s), 0) for s in D_minus_O)
        
        # 第三项：中间客户之间的边
        for h in range(1, k-1):
            lhs_minus += lp_sol.get((seq[h-1], seq[h]), 0)
        
        # 第四项：中间客户到i1的边
        for h in range(2, k):
            lhs_minus += 2 * lp_sol.get((seq[h], seq[0]), 0)
        
        # 检查是否违反D-k约束
        if lhs_minus > k:
            violation = {
                'sequence': list(seq),
                'depots_O': list(O),
                'depots_D_minus_O': list(D_minus_O),
                'lhs_value': lhs_minus,
                'inequality': f"D-{k}: sum(...) <= {k}"
            }
            violations.append(violation)
    
    # 对每个客户作为起始点，开始DFS扩展序列
    for start_customer in customers:
        dfs_extend_sequence([start_customer], [c for c in customers if c != start_customer], 1)

    return violations

if __name__ == '__main__':
    from instance import generate_instance_amdTSP
    from lp_solution import generate_lp_solution_amdTSP
    cm, D, V, max_routes = generate_instance_amdTSP()
    lp = generate_lp_solution_amdTSP(cm, D, V, max_routes)
    viols = separate_depot_fixing_amdTSP(lp, D, V)
    print("A-MDTSP Depot fixing violations:")
    for v in viols:
        print(v)