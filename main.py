# main.py (A-MDTSP)
from instance import generate_instance_amdTSP, generate_test_instances
from lp_solution import generate_lp_solution_amdTSP
from sep_dk_depot import separate_depot_fixing_amdTSP
from sep_comb import separate_comb
from data_storage import save_instances, load_all_instances
import numpy as np

import os
import json

def main():
    # 定义测试实例配置
    instance_sizes = [
        (2, 10),  # 小规模实例
        (3, 15),  # 中规模实例
        (4, 20),  # 大规模实例
    ]
    cost_types = [1, 2, 3]  # 三种不同类型的成本矩阵
    num_instances_per_config = 5  # 每个配置生成的实例数量
    
    # 生成所有测试实例
    instances = generate_test_instances(instance_sizes, cost_types, num_instances_per_config)
    
    # 保存测试实例，确保可复现性
    save_instances(instances)
    
    # 结果统计
    results = []
    total_instances = len(instances)
    
    print(f"\n=== 开始测试 {total_instances} 个实例 ===")
    
    for idx, instance in enumerate(instances, 1):
        print(f"\n处理实例 {idx}/{total_instances}: {instance['id']}")
        
        # 记录开始时间
        import time
        start_time = time.time()
        
        # 生成 LP 解
        lp_sol = generate_lp_solution_amdTSP(
            instance['cost_matrix'],
            instance['depots'],
            instance['customers'],
            instance['max_routes']
        )
        lower_bound = sum(lp_sol.values())
        
        # 分离 Depot Fixing 约束
        depot_violations = separate_depot_fixing_amdTSP(
            lp_sol,
            instance['depots'],
            instance['customers']
        )
        
        # 分离 Comb 不等式
        comb_violations = separate_comb(lp_sol, instance['customers'], instance['depots'])
        
        # 计算性能指标
        solve_time = time.time() - start_time
        nodes_explored = len(depot_violations) + len(comb_violations)
        
        # 记录结果
        result = {
            'instance_id': instance['id'],
            'num_depots': instance['num_depots'],
            'num_customers': instance['num_customers'],
            'cost_type': instance['cost_type'],
            'solve_time': solve_time,
            'nodes_explored': nodes_explored,
            'lower_bound': lower_bound,
            'depot_violations': len(depot_violations),
            'comb_violations': len(comb_violations)
        }
        results.append(result)
        
        # 输出当前实例结果
        print(f"求解时间: {solve_time:.3f} 秒")
        print(f"探索节点数: {nodes_explored}")
        print(f"下界值: {lower_bound:.2f}")
    
    # 保存实验结果
    results_dir = 'experiment_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 保存详细结果
    results_file = os.path.join(results_dir, 'detailed_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 生成结果表格
    print("\n=== 实验结果汇总 ===")
    print("\n按实例规模和成本类型统计平均性能:")
    
    # 按规模和类型分组统计
    for size in instance_sizes:
        num_d, num_c = size
        print(f"\n规模 D{num_d}C{num_c}:")
        print("类型  求解时间(秒)  节点数  下界值  Depot违反  Comb违反")
        print("-" * 55)
        
        for cost_type in cost_types:
            # 筛选当前配置的实例结果
            current_results = [
                r for r in results
                if r['num_depots'] == num_d
                and r['num_customers'] == num_c
                and r['cost_type'] == cost_type
            ]
            
            # 计算平均值
            avg_time = sum(r['solve_time'] for r in current_results) / len(current_results)
            avg_nodes = sum(r['nodes_explored'] for r in current_results) / len(current_results)
            avg_bound = sum(r['lower_bound'] for r in current_results) / len(current_results)
            avg_depot = sum(r['depot_violations'] for r in current_results) / len(current_results)
            avg_comb = sum(r['comb_violations'] for r in current_results) / len(current_results)
            
            print(f"{cost_type:^4d}  {avg_time:^11.3f}  {avg_nodes:^6.1f}  {avg_bound:^6.1f}  {avg_depot:^9.1f}  {avg_comb:^8.1f}")

if __name__ == '__main__':
    main()