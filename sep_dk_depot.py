# sep_dk_depot.py
import itertools

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
    
    # 扩展情况：检查更长的客户序列 (i1, i2, ..., ik)
    # 限制序列长度以控制计算复杂度
    for k in range(3, min(max_seq_length + 1, len(customers) + 1)):
        # 生成所有可能的k长度客户序列
        for seq in itertools.permutations(customers, k):
            # 对于每个序列，考虑不同的仓库子集O
            for o_size in range(1, len(depots)):
                for O in itertools.combinations(depots, o_size):
                    O = set(O)
                    D_minus_O = set(depots) - O
                    
                    # 计算D+k约束左端值
                    # 公式(30): sum_{s∈O} x_{i1,s} + sum_{s∈D\O} x_{s,ik} + sum_{h=2}^{k-1} x_{ih,i_{h-1}} + 2*sum_{h=2}^{k-1} x_{i1,ih}
                    
                    # 第一项：i1到O中仓库的边
                    lhs = sum(lp_sol.get((seq[0], s), 0) for s in O)
                    
                    # 第二项：D\O中仓库到ik的边
                    lhs += sum(lp_sol.get((s, seq[-1]), 0) for s in D_minus_O)
                    
                    # 第三项：中间客户之间的边
                    for h in range(1, k-1):
                        lhs += lp_sol.get((seq[h], seq[h-1]), 0)
                    
                    # 第四项：i1到中间客户的边
                    for h in range(2, k):
                        lhs += 2 * lp_sol.get((seq[0], seq[h]), 0)
                    
                    # 检查是否违反约束
                    if lhs > k:
                        violation = {
                            'sequence': list(seq),
                            'depots_O': list(O),
                            'depots_D_minus_O': list(D_minus_O),
                            'lhs_value': lhs,
                            'inequality': f"D+{k}: sum(...) <= {k}"
                        }
                        violations.append(violation)
                    
                    # 计算D-k约束左端值
                    # 公式(31): sum_{s∈O} x_{s,i1} + sum_{s∈D\O} x_{ik,s} + sum_{h=2}^{k-1} x_{i_{h-1},ih} + 2*sum_{h=2}^{k-1} x_{ih,i1}
                    
                    # 第一项：O中仓库到i1的边
                    lhs = sum(lp_sol.get((s, seq[0]), 0) for s in O)
                    
                    # 第二项：ik到D\O中仓库的边
                    lhs += sum(lp_sol.get((seq[-1], s), 0) for s in D_minus_O)
                    
                    # 第三项：中间客户之间的边
                    for h in range(1, k-1):
                        lhs += lp_sol.get((seq[h-1], seq[h]), 0)
                    
                    # 第四项：中间客户到i1的边
                    for h in range(2, k):
                        lhs += 2 * lp_sol.get((seq[h], seq[0]), 0)
                    
                    # 检查是否违反约束
                    if lhs > k:
                        violation = {
                            'sequence': list(seq),
                            'depots_O': list(O),
                            'depots_D_minus_O': list(D_minus_O),
                            'lhs_value': lhs,
                            'inequality': f"D-{k}: sum(...) <= {k}"
                        }
                        violations.append(violation)

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