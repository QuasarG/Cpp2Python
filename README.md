# A-MDTSP 问题求解器

## 项目简介

实现了一个用于求解非对称多仓库旅行商问题（Asymmetric Multi-Depot Traveling Salesman Problem, A-MDTSP）的Python程序。A-MDTSP是经典TSP问题的扩展变体，其特点是：

- 多个仓库作为起始点
- 非对称的距离/成本矩阵
- 每个仓库最多只能有一条巡回路径
- 所有客户点必须被访问一次

## 成本结构说明

本项目实现了三种不同类型的非对称成本结构，每种类型都具有其特定的特点和应用场景：

### 1. 完全随机非对称成本（Type I）

- **生成方式**：直接从均匀分布U(1, 1000)中随机生成整数
- **特点**：
  - 完全随机的非对称性
  - 无任何对称性基础
  - 无地理空间结构
- **应用场景**：适用于测试算法在最一般情况下的性能

### 2. 对称基础加扰动（Type II）

- **生成方式**：
  - 基础对称成本：a_ij = a_ji ~ U(1, 1000)
  - 非对称扰动：b_ij ~ U(1, 20)
  - 最终成本：c_ij = a_ij + b_ij
- **特点**：
  - 保持基本的对称性结构
  - 轻微的非对称扰动
  - 更接近实际运输场景
- **应用场景**：模拟现实中由于交通、时间等因素导致的轻微非对称性

### 3. 欧几里得距离加扰动（Type III）

- **生成方式**：
  - 基础成本：d_ij为节点在[0, 500]×[0, 500]网格中的欧几里得距离
  - 非对称扰动：b_ij ~ U(1, 20)
  - 最终成本：c_ij = d_ij + b_ij
- **特点**：
  - 具有明确的地理空间结构
  - 节点分布特征（一半均匀分布，一半聚类）
  - 适度的非对称扰动
- **应用场景**：适用于模拟真实地理环境下的运输问题

## 实验分析

### 实例规模设置

我们设计了三种不同规模的测试实例：

1. **小规模实例**：
   - 仓库数量：2
   - 客户数量：10
   - 适用于快速测试和调试

2. **中规模实例**：
   - 仓库数量：3
   - 客户数量：15
   - 用于验证算法稳定性

3. **大规模实例**：
   - 仓库数量：4
   - 客户数量：20
   - 测试算法的扩展性能

### 求解效果分析

针对不同成本类型的实例，我们的求解器表现如下：

1. **Type I（完全随机）实例**：
   - Depot Fixing约束效果显著
   - Comb不等式能有效收缩可行域
   - 整体求解时间适中

2. **Type II（对称基础）实例**：
   - 基于对称性的预处理效果好
   - 分离算法收敛较快
   - 解的质量较高

3. **Type III（欧几里得）实例**：
   - 利用空间结构特征
   - 聚类信息辅助分离
   - 计算效率最优

## 项目扩展指南（后续添加）

### Gurobi求解器集成

1. **环境配置**
   ```bash
   pip install gurobipy
   ```

2. **模型构建示例**
   ```python
   import gurobipy as gp
   
   def build_model(cost_matrix, depots, customers):
       model = gp.Model("A-MDTSP")
       # 添加决策变量
       x = model.addVars([(i,j) for i in nodes for j in nodes], 
                        vtype=GRB.BINARY, name="x")
       # 添加约束...
       return model
   ```

3. **求解参数调优**
   - 设置合适的时间限制
   - 配置预求解选项
   - 调整分支定界参数

### 添加新的切割平面

1. **创建分离器类**
   ```python
   class NewSeparator:
       def __init__(self, lp_sol, depots, customers):
           self.lp_sol = lp_sol
           self.depots = depots
           self.customers = customers
   
       def separate(self):
           # 实现分离逻辑
           pass
   ```

2. **集成到主程序**
   ```python
   # 在main.py中添加
   separator = NewSeparator(lp_sol, depots, customers)
   cuts = separator.separate()
   ```

## 测试用例管理

### 测试用例说明

#### 命名规则

测试用例文件采用统一的命名格式：`DxCyTz_n.json`，其中：
- `x`: 仓库数量（2-4）
- `y`: 客户数量（10-20）
- `z`: 成本类型（1-3）
- `n`: 实例编号（0-4）

例如：`D2C10T1_0.json`表示具有2个仓库、10个客户、Type I成本类型的第0号实例。

#### 文件结构

每个测试用例以JSON格式存储，包含以下字段：
```json
{
  "id": "D2C10T1_0",           // 实例唯一标识
  "num_depots": 2,            // 仓库数量
  "num_customers": 10,        // 客户数量
  "cost_type": 1,            // 成本类型（1-3）
  "seed": 204,               // 随机种子
  "cost_matrix": [...],      // 成本矩阵
  "depots": [0, 1],          // 仓库节点编号
  "customers": [2,3,...,11],  // 客户节点编号
  "max_routes": 1            // 每个仓库最大路径数
}
```

### 生成测试实例

```python
# 配置测试参数
instance_sizes = [
    (2, 10),  # 小规模
    (3, 15),  # 中规模
    (4, 20)   # 大规模
]
cost_types = [1, 2, 3]  # 三种成本类型
num_instances = 5  # 每种配置的实例数

# 生成实例
instances = generate_test_instances(
    instance_sizes,
    cost_types,
    num_instances
)
```

### 实例存储与复现

1. **存储格式**：
   - JSON格式保存实例数据
   - 包含完整的问题参数
   - 记录随机种子确保可复现

2. **数据结构**：
   ```python
   instance = {
       'id': 'unique_instance_id',
       'num_depots': num_depots,
   }
   ```

## 项目结构说明

### 核心模块

1. **instance.py**
   - 负责生成A-MDTSP问题的测试实例
   - 实现了三种不同类型的成本矩阵生成方法
   - 提供实例参数的随机生成功能

2. **lp_solution.py**
   - 实现线性规划(LP)解的生成
   - 构建A-MDTSP的数学模型
   - 调用优化求解器获取LP解

3. **sep_dk_depot.py**
   - 实现Depot Fixing约束的分离算法
   - 检测和生成违反depot约束的不等式
   - 优化求解过程中的depot分配

4. **sep_comb.py**
   - 实现Comb不等式的分离算法
   - 检测和生成违反comb结构的不等式
   - 提升解的质量

### 数据管理

1. **test_instances/**
   - 存储生成的测试实例
   - 采用统一的命名规范
   - JSON格式保存实例数据

2. **experiment_results/**
   - 存储实验结果和性能数据
   - 包含详细的求解统计信息
   - JSON格式保存结果数据

### 主程序

**main.py**
- 程序的入口点
- 实现完整的求解流程
- 包含实验配置和结果统计
- 提供结果可视化和分析功能

### 项目依赖

**requirements.txt**
- numpy：用于数值计算和矩阵操作
- gurobipy（可选）：优化求解器，用于扩展搜索空间
- matplotlib（可选）：用于结果可视化
- pandas（可选）：用于数据处理和分析
       'num_customers': num_customers,
       'cost_type': cost_type,
       'seed': seed,
       'cost_matrix': cost_matrix.tolist(),
       'depots': depots,
       'customers': customers
   }
   ```

3. **加载与验证**：
   ```python
   # 加载保存的实例
   loaded_instances = load_all_instances()
   
   # 验证实例正确性
   for instance in loaded_instances:
       verify_instance(instance)
   ```

## 使用方法

### 环境要求

- Python 3.x
- NumPy库

### 安装依赖

```bash
pip install numpy
```

### 运行程序

1. 直接运行主程序：
```bash
python main.py
```

2. 分别测试各个模块：
```bash
python instance.py      # 测试实例生成
python lp_solution.py   # 测试LP解生成
python sep_dk_depot.py  # 测试Depot Fixing分离
python sep_comb.py      # 测试Comb不等式分离
```

## 注意事项

1. 确保正确配置Python环境和依赖包
2. 大规模实例计算可能需要较长时间
3. 建议先使用小规模实例进行测试
4. 定期保存实验结果和中间数据