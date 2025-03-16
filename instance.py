# instance.py
import numpy as np

def generate_instance_amdTSP(num_depots=2, num_customers=4, cost_type=1, seed=42, asymmetry_factor=0.2):
    """
    生成A-MDTSP问题实例
    
    参数:
        num_depots (int): 仓库数量
        num_customers (int): 客户数量
        cost_type (int): 成本矩阵类型
            1: 完全随机成本 c_ij ~ U(1, 1000)
            2: 欧几里得距离 + 随机噪声
            3: 聚类结构 + 非对称扰动
        seed (int): 随机数种子
        asymmetry_factor (float): 非对称性因子，控制成本矩阵的非对称程度
    """
    np.random.seed(seed)
    total_nodes = num_depots + num_customers
    
    # 生成节点坐标（用于类型2）
    coords = np.random.rand(total_nodes, 2) * 1000
    
    # 根据类型生成成本矩阵
    if cost_type == 1:
        # 类型I：完全随机非对称成本
        base_cost = np.random.randint(1, 1001, size=(total_nodes, total_nodes)).astype(float)
        asymmetric_noise = np.random.uniform(-asymmetry_factor, asymmetry_factor, size=(total_nodes, total_nodes)) * base_cost
        cost_matrix = base_cost + asymmetric_noise
    elif cost_type == 2:
        # 类型II：欧几里得距离 + 非对称扰动
        # 生成节点坐标
        coords = np.random.uniform(0, 1000, size=(total_nodes, 2))
        # 计算欧几里得距离
        cost_matrix = np.zeros((total_nodes, total_nodes))
        for i in range(total_nodes):
            for j in range(total_nodes):
                if i != j:
                    cost_matrix[i,j] = np.sqrt(np.sum((coords[i] - coords[j])**2))
        # 添加非对称扰动
        asymmetric_noise = np.random.uniform(-asymmetry_factor, asymmetry_factor, size=(total_nodes, total_nodes)) * cost_matrix
        cost_matrix = cost_matrix + asymmetric_noise
    else:
        # 类型III：欧几里得距离 + 非对称扰动
        # 生成节点坐标，一半均匀分布，一半聚类
        coords = np.zeros((total_nodes, 2))
        # 前一半节点均匀分布
        half_nodes = total_nodes // 2
        coords[:half_nodes] = np.random.uniform(0, 500, size=(half_nodes, 2))
        # 后一半节点形成聚类（这里简单示例使用2个聚类中心）
        centers = np.random.uniform(0, 500, size=(2, 2))
        for i in range(half_nodes, total_nodes):
            center_idx = np.random.randint(2)
            coords[i] = centers[center_idx] + np.random.normal(0, 50, 2)
            coords[i] = np.clip(coords[i], 0, 500)  # 确保在边界内
        
        # 计算欧几里得距离
        cost_matrix = np.zeros((total_nodes, total_nodes))
        for i in range(total_nodes):
            for j in range(total_nodes):
                if i != j:
                    cost_matrix[i,j] = np.floor(np.sqrt(np.sum((coords[i] - coords[j])**2)))
        
        # 添加非对称扰动
        perturbation = np.random.randint(1, 21, size=(total_nodes, total_nodes))
        cost_matrix = cost_matrix + perturbation
    
    np.fill_diagonal(cost_matrix, np.inf)  # 防止自环

    # 仓库集合 D 和客户集合 V
    depots = list(range(num_depots))
    customers = list(range(num_depots, total_nodes))

    # 设置 A-MDTSP 限制：每个仓库最多只能有一个巡回路径
    max_routes_per_depot = 1  # 关键参数，A-MDTSP 限制

    return cost_matrix, depots, customers, max_routes_per_depot

def generate_test_instances(instance_sizes, cost_types, num_instances_per_config=5):
    """
    批量生成测试实例
    
    参数:
        instance_sizes: 列表，每个元素是(num_depots, num_customers)元组
        cost_types: 列表，包含要测试的成本矩阵类型
        num_instances_per_config: 每个配置生成的实例数量
    
    返回:
        instances: 列表，包含所有生成的实例信息
    """
    instances = []
    for size in instance_sizes:
        num_depots, num_customers = size
        for cost_type in cost_types:
            for instance_id in range(num_instances_per_config):
                seed = instance_id + hash((num_depots, num_customers, cost_type)) % 10000
                instance = {
                    'id': f"D{num_depots}C{num_customers}T{cost_type}_{instance_id}",
                    'num_depots': num_depots,
                    'num_customers': num_customers,
                    'cost_type': cost_type,
                    'seed': seed
                }
                cost_matrix, depots, customers, max_routes = generate_instance_amdTSP(
                    num_depots=num_depots,
                    num_customers=num_customers,
                    cost_type=cost_type,
                    seed=seed
                )
                instance['cost_matrix'] = cost_matrix
                instance['depots'] = depots
                instance['customers'] = customers
                instance['max_routes'] = max_routes
                instances.append(instance)
    return instances

if __name__ == '__main__':
    # 测试实例生成
    instance_sizes = [(2, 10), (3, 15), (4, 20)]
    cost_types = [1, 2, 3]
    instances = generate_test_instances(instance_sizes, cost_types, num_instances_per_config=2)
    
    print(f"生成了 {len(instances)} 个测试实例")
    for instance in instances[:2]:  # 只显示前两个实例的信息
        print(f"\n实例 {instance['id']}:")
        print(f"- 仓库数量: {instance['num_depots']}")
        print(f"- 客户数量: {instance['num_customers']}")
        print(f"- 成本类型: {instance['cost_type']}")
        print(f"- 成本矩阵形状: {instance['cost_matrix'].shape}")