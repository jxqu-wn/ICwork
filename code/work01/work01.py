import pandas as pd
from galogic import *
from tqdm import tqdm
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 读取文件
    filename = '../../source/1 GA/全国文保单位.xlsx'

    data = pd.read_excel(filename)
    data = data[['lat', 'lon']]

    # 加载坐标点信息
    for i in range(len(data)):
        RouteManager.addDustbin(Dustbin(x=data.iat[i, 0], y=data.iat[i, 1]))

    # 初始化种群
    pop = Population(50, True)
    print('Initial distance: ' + str(pop.getFittest().getDistance()))
    yaxis = []
    xaxis = []
    for i in tqdm(range(5)):
        pop = GA.evolvePopulation(pop)
        fittest = pop.getFittest().getDistance()
        yaxis.append(fittest)
        xaxis.append(i)
        if fittest < 1050:
            break
    print('Final distance: ' + str(fittest))
    print('Final Route: ' + pop.getFittest().toString())

    fig = plt.figure()
    # plt.ylim(0, 80000)
    plt.plot(xaxis, yaxis, 'r-')
    plt.show()
