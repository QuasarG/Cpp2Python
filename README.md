# A-MDTSP 问题求解器

## 项目简介

实现了一个用于求解非对称多仓库旅行商问题（Asymmetric Multi-Depot Traveling Salesman Problem, A-MDTSP）的Python程序。A-MDTSP是经典TSP问题的扩展变体，其特点是：

- 多个仓库作为起始点
- 非对称的距离/成本矩阵
- 每个仓库最多只能有一条巡回路径
- 所有客户点必须被访问一次

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

## 环境要求与安装

### 项目依赖

**requirements.txt**
- numpy：用于数值计算和矩阵操作
- gurobipy（可选）：优化求解器，用于扩展搜索空间
- matplotlib（可选）：用于结果可视化
- pandas（可选）：用于数据处理和分析

### 安装依赖

```bash
pip install numpy
```

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

### 使用方法

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

## 项目扩展指南

### 扩展到其他问题变体

本项目目前专注于A-MDTSP问题，但框架设计具有良好的扩展性，可以方便地扩展到其他相关问题变体。以下是扩展到几种常见变体的具体方法：

#### 1. 扩展到A-MDmTSP（多车辆非对称多仓库TSP）

- **修改instance.py**
  ```python
  # 添加车辆数量参数
  def generate_instance(num_depots, num_customers, cost_type, num_vehicles_per_depot=1):
      instance = {...}
      instance['num_vehicles_per_depot'] = num_vehicles_per_depot
      return instance
  ```

- **扩展lp_solution.py**
  ```python
  # 修改决策变量定义
  # x_{i,j,k} 表示车辆k是否使用边(i,j)
  x = {}
  for k in range(num_vehicles):
      for i in nodes:
          for j in nodes:
              if i != j:
                  x[i,j,k] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")
  
  # 添加车辆分配约束
  for i in customers:
      model.addConstr(gp.quicksum(x[i,j,k] for j in nodes if j != i for k in range(num_vehicles)) == 1)
  ```

#### 2. 扩展到A-MDCVRP（非对称多仓库容量限制车辆路径问题）

- **修改instance.py**
  ```python
  def generate_instance(num_depots, num_customers, cost_type, vehicle_capacity=100):
      instance = {...}
      # 添加客户需求量
      instance['demands'] = {i: random.randint(1, 20) for i in instance['customers']}
      # 添加车辆容量
      instance['vehicle_capacity'] = vehicle_capacity
      return instance
  ```

- **扩展lp_solution.py**
  ```python
  # 添加容量约束
  for k in range(num_vehicles):
      model.addConstr(
          gp.quicksum(demands[i] * gp.quicksum(x[i,j,k] for j in nodes if j != i) 
                     for i in customers) <= vehicle_capacity
      )
  ```

- **添加新的分离算法**
  ```python
  # 创建新文件 sep_capacity.py
  class CapacityCutSeparator:
      def __init__(self, lp_sol, depots, customers, demands, capacity):
          self.lp_sol = lp_sol
          self.depots = depots
          self.customers = customers
          self.demands = demands
          self.capacity = capacity
      
      def separate(self):
          # 实现容量约束分离算法
          # 例如：分离rounded capacity cuts
          cuts = []
          # ...
          return cuts
  ```

#### 3. 扩展到A-CLRP（非对称容量限制选址路径问题）

- **修改instance.py**
  ```python
  def generate_instance(num_potential_depots, num_customers, cost_type):
      instance = {...}
      # 添加仓库开设成本
      instance['depot_costs'] = {i: random.randint(500, 2000) for i in instance['depots']}
      # 添加仓库容量
      instance['depot_capacities'] = {i: random.randint(50, 150) for i in instance['depots']}
      return instance
  ```

- **扩展lp_solution.py**
  ```python
  # 添加仓库选址决策变量
  y = {}
  for i in depots:
      y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")
  
  # 添加仓库开设成本到目标函数
  obj = gp.quicksum(cost_matrix[i][j] * x[i,j,k] for i in nodes for j in nodes if i != j for k in range(num_vehicles))
  obj += gp.quicksum(depot_costs[i] * y[i] for i in depots)
  model.setObjective(obj, GRB.MINIMIZE)
  
  # 添加仓库容量约束
  for i in depots:
      model.addConstr(
          gp.quicksum(demands[j] * gp.quicksum(x[i,j,k] for k in range(num_vehicles)) for j in customers) <= 
          depot_capacities[i] * y[i]
      )
  ```

- **修改main.py的求解流程**
  ```python
  # 添加仓库选址相关处理
  def solve_clrp(instance):
      # 初始化模型
      model = build_model(instance)
      
      # 迭代求解
      while True:
          # 求解当前模型
          model.optimize()
          
          # 分离违反的约束
          depot_cuts = separate_depot_constraints(model)
          capacity_cuts = separate_capacity_constraints(model)
          comb_cuts = separate_comb_inequalities(model)
          
          # 添加切割平面
          if not (depot_cuts or capacity_cuts or comb_cuts):
              break
              
          # 添加所有找到的切割平面
          for cut in depot_cuts + capacity_cuts + comb_cuts:
              model.addConstr(cut)
      
      # 返回最终解
      return extract_solution(model)
  ```

### 实现通用框架

为了更好地支持多种问题变体，可以实现一个通用的求解框架：

```python
# 在data_storage.py中添加
class ProblemType(Enum):
    AMDTSP = "A-MDTSP"    # 非对称多仓库TSP
    AMDmTSP = "A-MDmTSP"  # 非对称多车辆多仓库TSP
    AMDCVRP = "A-MDCVRP"  # 非对称多仓库容量限制VRP
    ACLRP = "A-CLRP"      # 非对称容量限制选址路径问题

# 在main.py中实现通用求解器
def solve_problem(instance, problem_type):
    # 根据问题类型选择相应的模型构建和求解方法
    if problem_type == ProblemType.AMDTSP:
        return solve_amdtsp(instance)
    elif problem_type == ProblemType.AMDmTSP:
        return solve_amdmtsp(instance)
    elif problem_type == ProblemType.AMDCVRP:
        return solve_amdcvrp(instance)
    elif problem_type == ProblemType.ACLRP:
        return solve_aclrp(instance)
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")
```

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

## 注意事项

1. 确保正确配置Python环境和依赖包
2. 大规模实例计算可能需要较长时间
3. 建议先使用小规模实例进行测试
4. 定期保存实验结果和中间数据