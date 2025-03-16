# sep_dk_depot.py

def separate_depot_fixing_amdTSP(lp_sol, depots, customers):
    violations = []
    
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