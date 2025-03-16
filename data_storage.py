# data_storage.py
import json
import os
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于处理NumPy数组"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_instances(instances, output_dir='test_instances'):
    """保存测试实例到JSON文件
    
    参数:
        instances: 测试实例列表
        output_dir: 输出目录名
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 按规模和类型分组保存
    for instance in instances:
        # 构造文件名
        filename = f"{instance['id']}.json"
        filepath = os.path.join(output_dir, filename)
        
        # 保存实例数据
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(instance, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)

def load_instance(instance_id, input_dir='test_instances'):
    """根据实例ID加载特定的测试实例
    
    参数:
        instance_id: 实例ID (例如: 'D2C10T1_0')
        input_dir: 输入目录名
    
    返回:
        加载的测试实例字典
    """
    filepath = os.path.join(input_dir, f"{instance_id}.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到实例文件: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        instance = json.load(f)
    
    # 将列表转回NumPy数组
    instance['cost_matrix'] = np.array(instance['cost_matrix'])
    return instance

def load_all_instances(input_dir='test_instances'):
    """加载目录下的所有测试实例
    
    参数:
        input_dir: 输入目录名
    
    返回:
        测试实例列表
    """
    if not os.path.exists(input_dir):
        return []
    
    instances = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            instance_id = filename[:-5]  # 移除.json后缀
            try:
                instance = load_instance(instance_id, input_dir)
                instances.append(instance)
            except Exception as e:
                print(f"加载实例 {instance_id} 时出错: {e}")
    
    return instances

def filter_instances(instances, num_depots=None, num_customers=None, cost_type=None):
    """根据条件筛选测试实例
    
    参数:
        instances: 测试实例列表
        num_depots: 仓库数量
        num_customers: 客户数量
        cost_type: 成本类型
    
    返回:
        符合条件的测试实例列表
    """
    filtered = instances
    
    if num_depots is not None:
        filtered = [inst for inst in filtered if inst['num_depots'] == num_depots]
    
    if num_customers is not None:
        filtered = [inst for inst in filtered if inst['num_customers'] == num_customers]
    
    if cost_type is not None:
        filtered = [inst for inst in filtered if inst['cost_type'] == cost_type]
    
    return filtered