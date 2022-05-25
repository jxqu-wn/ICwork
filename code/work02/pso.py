import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm
from pylab import mpl


targetx = np.array([0.0, 4.0])
targety = np.array([0.0, 6.0])
obstaclex = np.array([1.2, 1.5, 4])
obstacley = np.array([1.5, 4.5, 3])
obstacler = np.array([0.8, 1.5, 1.0])
popularity = 200  # 粒子群数量
maxspeed = 3.0  # 速度限制
iterate = 400  # 迭代次数


# 三次样条插值，代码原型来源于网络
class SPLINE(object):
    def __init__(self, p, p_):
        self.p = p  # 型值点
        self.p_ = p_  # 端点矢切
        self.n = len(p)  # 型值点数
        self.s = self.n - 1  # 线段数
        self.t = np.zeros(self.s)  # 线段长度
        self.m = np.zeros((self.n, self.n))  # 计算用矩阵
        self.C_y = []
        self.A = []
        self.B = []
        self.C = []
        self.D = []
        self.samples_t = []
        self.samples_s = []

        self.caculate_t()
        self.caculate_matrix()
        self.caculate_C_y()
        self.caculate_vector_cut()
        self.caculate_parameter()
        self.grasp_sample()

    def caculate_t(self):
        # 线段数量
        # self.s = len(self.p)-1
        # self.t = np.zeros(self.s)
        for i in range(self.s):
            self.t[i] = np.sqrt((self.p[i+1][0]-self.p[i][0])**2 + (self.p[i+1][1]-self.p[i][1])**2)

        return self.t

    def caculate_matrix(self):
        # t = SPLINE.caculate_t(p)
        # n = len(p)
        # self.m = np.zeros((n, n))
        for i in range(self.n):
            for j in range(self.n):
                if i == 0 and j == 0 or i == self.n-1 and j == self.n-1:
                    self.m[i][j] = 1
                elif i == 0 and j == 1 or i == self.n-1 and j == self.n-2:
                    self.m[i][j] = 0
                elif i == j:  # i不可能等于0
                    self.m[i][j] = 2*(self.t[i]+self.t[i-1])
                elif i+1 == j:  # i不可能到n-1
                    self.m[i][j] = self.t[i-1]
                elif i-1 == j:  # i不可能等于0
                    self.m[i][j] = self.t[i]
        return self.m

    def caculate_C_y(self):
        # n = len(p)
        # C = np.zeros(n)
        for i in range(self.n):
            if i == 0:
                # self.C_y[i] = self.p_[i]
                self.C_y.append(self.p_[0])
            elif i == self.n-1:
                # self.C_y[i] = self.p_[self.n-1]
                self.C_y.append(self.p_[1])
            else:
                temp = self.t[i-1] * self.t[i] * \
                (3 * (self.p[i+1] - self.p[i]) / self.t[i] ** 2 + 3 * (self.p[i] - self.p[i-1]) / self.t[i-1] ** 2)
                self.C_y.append(temp)
        self.C_y = np.array(self.C_y)
        return self.C_y

    def caculate_vector_cut(self):
        temp = np.linalg.solve(self.m, self.C_y)
        self.p_ = np.array(temp)
        return temp

    def caculate_parameter(self):
        for i in range(self.s):
            A = 2*(self.p[i] - self.p[i + 1]) / self.t[i] ** 3 + (self.p_[i + 1] + self.p_[i]) / self.t[i] ** 2
            self.A.append(A)
            B = 3*(self.p[i+1]-self.p[i])/self.t[i]**2 - (2*self.p_[i] + self.p_[i+1])/self.t[i]
            self.B.append(B)
            C = self.p_[i]
            self.C.append(C)
            D = self.p[i]
            self.D.append(D)
        self.A = np.array(self.A)
        self.B = np.array(self.B)
        self.C = np.array(self.C)
        self.D = np.array(self.D)
        return self.A, self.B, self.C, self.D

    # def caculate_S_t(self, i, t):
    #     return self.A[i] * t**3 + self.B[i] * t**2 + self.C[i]*t + self.D[i]

    def grasp_sample(self):
        for i in range(self.s):
            sample_t = np.arange(0, self.t[i], 0.1)
            sample_s = []
            for t in sample_t:
                S_t = self.A[i] * t**3 + self.B[i] * t**2 + self.C[i]*t + self.D[i]
                sample_s.append(S_t)
            # sample_y = calculate(result[(i - 1) * 4:i * 4], sample_x)
            self.samples_t.extend(sample_t)
            self.samples_s.extend(sample_s)
        self.samples_t.append(self.t[-1])
        s_t = self.A[-1] * self.t[-1]**3 + self.B[-1] * self.t[-1]**2 + self.C[-1]*self.t[-1] + self.D[-1]
        self.samples_s.append(s_t)
        self.samples_t = np.array(self.samples_t)
        self.samples_s = np.array(self.samples_s)

    def show_view(self):
        plt.plot(self.samples_s[:,0], self.samples_s[:,1], label="拟合曲线", color="black")
        plt.scatter(self.p[:,0], self.p[:,1], label="离散数据", color="red")
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        mpl.rcParams['axes.unicode_minus'] = False
        plt.title("三次样条函数")
        plt.legend(loc="upper left")
        plt.show()

    def __str__(self):
        return "型值点数={}\n线段数={}\n型值点={}\n端点矢切={}\n线段长度={}\n求矢切的矩阵={}\n求矢切的C={}\n" \
               "参数A={}\n参数B={}\n参数C={}\n参数D={}\n绘图t={}\n绘图点={}\n".format(self.n, self.s, self.p, self.p_,
                self.t, self.m, self.C_y, self.A, self.B, self.C, self.D, self.samples_t, self.samples_s)

    __repr__ = __str__


class Particle:
    def __init__(self):
        # 关键位置
        self.x = []
        self.y = []
        for i in range(3):
            self.x.append((np.random.uniform() - 0.5) * 20)
            self.y.append((np.random.uniform() - 0.5) * 20)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        # 速度
        self.vx = np.zeros(3)
        self.vy = np.zeros(3)
        # 插值结果
        self.allx = []
        self.ally = []
        self.solution()
        # 适应度值
        self.fitness = 0.0
        self.cost()
        # 粒子历史最优
        self.pbest = np.array([[self.x], [self.y], [self.fitness], [self.allx], [self.ally]])

    # solution
    def solution(self):
        # 三次样条插值
        x = np.array([targetx[0], self.x[0], self.x[1], self.x[2], targetx[1]])
        y = np.array([targety[0], self.y[0], self.y[1], self.y[2], targety[1]])
        new_x = []
        for i in range(3):
            new_x = np.append(new_x, np.arange(x[i], x[i+1], 0.1))
        p = np.array([x, y]).T
        p_ = np.array([[0.0, 0.0], [0.0, 0.0]])
        spline = SPLINE(p, p_)
        newp = spline.samples_s.T
        self.allx = newp[0]
        self.ally = newp[1]

    # 计算适应度值
    def cost(self):
        distance = 0.0
        length = self.allx.__len__()
        for i in range(length - 1):
            distance = distance + \
                       np.sqrt(np.square(self.allx[i] - self.allx[i + 1]) + np.square(self.ally[i] - self.ally[i + 1]))
        violation = 0.0
        for i in range(length):
            for j in range(3):
                violation = violation + \
                            max(1 - np.sqrt(
                                np.square(self.allx[i] - obstaclex[j]) + np.square(self.ally[i] - obstacley[j])) /
                                obstacler[j], 0.0)
        violation = violation / length
        self.fitness = distance * (1 + 100 * violation)

    # 绘图
    def draw_best(self):
        plt.plot(self.pbest[3][0], self.pbest[4][0])
        plt.plot(self.pbest[0][0], self.pbest[1][0], 'o')
        circle1 = plt.Circle(xy=(1.2, 1.5), radius=0.8)
        circle2 = plt.Circle(xy=(1.5, 4.5), radius=1.5)
        circle3 = plt.Circle(xy=(4.0, 3.0), radius=1.0)
        plt.gcf().gca().add_artist(circle1)
        plt.gcf().gca().add_artist(circle2)
        plt.gcf().gca().add_artist(circle3)

        plt.xlim(-2, 8)
        plt.ylim(-2, 8)
        plt.show()

    def draw(self):
        plt.plot(self.allx, self.ally)
        plt.show()


class PSO:
    def __init__(self):
        # 初始化粒子群
        self.particles = [Particle() for i in range(popularity)]
        # 初始化全局最优
        self.index = 0
        self.gbest = []
        self.findBest()

    def update(self):
        for i in range(popularity):
            # 更新速度
            vx = self.particles[i].vx + \
                 2 * np.random.uniform() * (self.particles[i].pbest[0][0] - self.particles[i].x) + \
                 2 * np.random.uniform() * (self.gbest[0][0] - self.particles[i].x)
            vy = self.particles[i].vy + \
                 2 * np.random.uniform() * (self.particles[i].pbest[1][0] - self.particles[i].y) + \
                 2 * np.random.uniform() * (self.gbest[1][0] - self.particles[i].y)
            for k in range(3):
                if vx[k] > maxspeed:
                    self.particles[i].vx[k] = maxspeed
                elif vx[k] < -maxspeed:
                    self.particles[i].vx[k] = -maxspeed
                else:
                    self.particles[i].vx[k] = np.copy(vx[k])
                if vy[k] > maxspeed:
                    self.particles[i].vy[k] = maxspeed
                elif vy[k] < -maxspeed:
                    self.particles[i].vy[k] = -maxspeed
                else:
                    self.particles[i].vy[k] = np.copy(vy[k])
            # 更新位置
            x = self.particles[i].x + self.particles[i].vx
            y = self.particles[i].y + self.particles[i].vy
            for l in range(3):
                if x[l] > 10.0:
                    self.particles[i].x[l] = 10.0
                elif x[l] < -10.0:
                    self.particles[i].x[l] = -10.0
                else:
                    self.particles[i].x[l] = x[l]
                if y[l] > 10.0:
                    self.particles[i].y[l] = 10.0
                elif y[l] < -10.0:
                    self.particles[i].y[l] = -10.0
                else:
                    self.particles[i].y[l] = y[l]
            # 更新插值路线
            self.particles[i].solution()
            # 更新适应度值
            self.particles[i].cost()
            # 更新粒子最优
            if self.particles[i].fitness < self.particles[i].pbest[2][0]:
                self.particles[i].pbest = np.array([
                    [np.copy(self.particles[i].x)], [np.copy(self.particles[i].y)], [self.particles[i].fitness],
                    [self.particles[i].allx], [self.particles[i].ally]
                ])
        # 更新全局最优
        self.findBest()

    def findBest(self):
        allcost = []
        for i in range(popularity):
            allcost.append(self.particles[i].pbest[2][0])
        allcost = np.array(allcost)
        self.index = np.min(np.argmin(allcost, axis=0))
        self.gbest = self.particles[self.index].pbest

    def draw(self):
        self.particles[self.index].draw_best()


if __name__ == '__main__':
    pso = PSO()
    bestlist = []
    bestlist.append(pso.gbest[2][0])
    for i in tqdm(range(iterate)):
        pso.update()
        bestlist.append(pso.gbest[2][0])
    pso.draw()
    time = np.arange(iterate + 1)
    plt.plot(time, bestlist)
    plt.show()
