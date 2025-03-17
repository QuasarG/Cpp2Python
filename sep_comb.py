# sep_comb.py

def find_weakly_connected_components(lp_sol, nodes, threshold=0.5):
    """
    查找LP解中的弱连通分量
    
    参数:
        lp_sol: LP解，字典，键为边(i,j)，值为对应的解值x_ij
        nodes: 节点集合
        threshold: 边权重阈值，大于此值的边被认为是连通的
    
    返回:
        components: 列表，包含所有弱连通分量（每个分量是节点集合）
    """
    # 构建无向图（忽略边的方向）
    graph = {}
    for i in nodes:
        graph[i] = set()
    
    # 添加边（只考虑权重大于阈值的边）
    for (i, j), weight in lp_sol.items():
        if i in nodes and j in nodes and weight >= threshold:
            graph[i].add(j)
            if j not in graph:  # 确保j也是图中的节点
                graph[j] = set()
            graph[j].add(i)
    
    # 使用DFS查找连通分量
    visited = set()
    components = []
    
    def dfs(node, component):
        visited.add(node)
        component.add(node)
        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                dfs(neighbor, component)
    
    # 对每个未访问的节点运行DFS
    for node in nodes:
        if node not in visited:
            component = set()
            dfs(node, component)
            components.append(component)
    
    return components

def separate_comb(lp_sol, customers, depots=None, violation_threshold=2.0):
    """
    实现Comb不等式的分离算法，使用贪婪策略构造handle和tooth
    
    参数:
        lp_sol: LP解，字典，键为边(i,j)，值为对应的解值x_ij
        customers: 客户集合
        depots: 仓库集合，如果提供则考虑包含仓库的齿
        violation_threshold: 违反阈值，默认为2.0（ATSP标准）
    
    返回:
        violations: 列表，包含所有违反的不等式信息
    """
    violations = []
    all_nodes = set(customers)
    if depots:
        all_nodes.update(depots)
    
    # 查找弱连通分量
    components = find_weakly_connected_components(lp_sol, all_nodes)
    
    # 对每个连通分量尝试构造Comb不等式
    for component in components:
        if len(component) < 3:  # 至少需要3个节点才能形成有意义的Comb
            continue
        
        # 尝试不同的初始handle
        for start_node in component:
            # 初始化handle为单个节点
            H = {start_node}
            
            # 贪婪扩展handle：添加节点使x(δ+(H))最小
            while len(H) < len(component) - 1:  # 保留至少一个节点用于tooth
                candidates = component - H
                best_candidate = None
                best_delta = float('inf')
                
                for c in candidates:
                    new_H = H | {c}
                    new_delta = sum(lp_sol.get((i, j), 0) for i in new_H for j in lp_sol if j not in new_H)
                    if new_delta < best_delta:
                        best_candidate = c
                        best_delta = new_delta
                
                if best_candidate:
                    H.add(best_candidate)
                else:
                    break
            
            # 构造tooth：从handle外的节点开始
            remaining = component - H
            if not remaining:  # 确保有剩余节点用于tooth
                continue
            
            # 如果有仓库，优先选择仓库作为tooth的一部分
            T = set()
            if depots:
                for d in depots:
                    if d in remaining:
                        T.add(d)
                        break
            
            # 如果没有仓库或没有选择到仓库，随机选择一个起始节点
            if not T:
                import random
                T.add(random.choice(list(remaining)))
            
            # 贪婪扩展tooth：添加节点使x(δ+(T))最小
            while T and len(T) < len(remaining):
                candidates = remaining - T
                best_candidate = None
                best_delta = float('inf')
                
                for c in candidates:
                    new_T = T | {c}
                    new_delta = sum(lp_sol.get((i, j), 0) for i in new_T for j in lp_sol if j not in new_T)
                    if new_delta < best_delta:
                        best_candidate = c
                        best_delta = new_delta
                
                if best_candidate:
                    T.add(best_candidate)
                else:
                    break
            
            # 计算Comb不等式的左端值
            H_out_sum = sum(lp_sol.get((i, j), 0) for i in H for j in lp_sol if j not in H)
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