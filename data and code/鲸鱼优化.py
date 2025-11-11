import numpy as np

class WhaleOptimizationAlgorithm:
    def __init__(self, fitness_function, num_whales, num_iterations, dim, lb, ub):
        """
        WOA算法
        :param fitness_function: 目标函数
        :param num_whales: 种群数量
        :param num_iterations: 最大迭代次数
        :param dim: 解的维度
        :param lb: 每个维度的下界
        :param ub: 每个维度的上界
        """
        self.fitness_function = fitness_function
        self.num_whales = num_whales
        self.num_iterations = num_iterations
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.positions = np.random.uniform(lb, ub, (num_whales, dim))  # 初始化鲸鱼位置
        self.best_position = None  # 最优位置
        self.best_fitness = float('inf')  # 最优适应度

    def update_position(self, whale_position, a, b, c, best_position):
        """
        根据鲸鱼优化算法的螺旋更新公式更新鲸鱼位置
        :param whale_position: 当前鲸鱼的位置
        :param a, b, c: 用于更新的参数
        :param best_position: 最优位置
        :return: 更新后的位置
        """
        # 更新位置的螺旋公式（围绕猎物运动）
        r = np.linalg.norm(best_position - whale_position)  # 当前位置与最优解的距离
        A = 2 * a * np.random.random() - a  # 随机的系数
        C = 2 * np.random.random()  # 随机的系数

        # 更新位置的螺旋公式
        D = np.abs(C * best_position - whale_position)  # 猎物的位置和当前个体的位置差异
        new_position = best_position - A * D  # 计算新位置

        # 限制新位置的范围
        new_position = np.clip(new_position, self.lb, self.ub)

        return new_position

    def optimize(self):
        """
        执行鲸鱼优化算法
        :return: 最优解
        """
        for t in range(self.num_iterations):
            a = 2 - t * (2 / self.num_iterations)  # 线性递减的a值，控制探索和开发
            for i in range(self.num_whales):
                whale_position = self.positions[i]
                fitness = self.fitness_function(whale_position)

                # 更新最优解
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_position = whale_position

                # 更新位置
                self.positions[i] = self.update_position(whale_position, a, 1, 1, self.best_position)

            print(f"Iteration {t+1}/{self.num_iterations}, Best Fitness: {self.best_fitness}")

        return self.best_position, self.best_fitness

# 示例：一个简单的目标函数
def fitness_function(position):
    # 一个简单的Rastrigin函数（通常用于测试优化算法）
    A = 10
    return A * len(position) + sum(x**2 - A * np.cos(2 * np.pi * x) for x in position)

# 参数设置
num_whales = 30  # 种群数量
num_iterations = 100  # 最大迭代次数
dim = 5  # 问题维度
lb = -5.12  # 变量下界
ub = 5.12  # 变量上界

# 初始化并运行WOA
woa = WhaleOptimizationAlgorithm(fitness_function, num_whales, num_iterations, dim, lb, ub)
best_position, best_fitness = woa.optimize()

print("Best Position:", best_position)
print("Best Fitness:", best_fitness)
