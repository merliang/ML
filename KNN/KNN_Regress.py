import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

data = pd.read_csv(r"C:\\Users\\不归客\Desktop\\iris.csv", header=0)
# 删除不需要的Id与Species列
data.drop(["Id", "Species"], axis=1, inplace=True)
# 删除重复数据
data.drop_duplicates(inplace=True)

# KNN回归


class KNN:
    '''
    使用python实现KNN算法(回归预测)
    该算法用于回归预测，根据前3个特征属性，寻找最近的k个邻居，然后再根据k个邻居的第4个特征属性，去预测当前样本的第4个特征值
    '''

    def __init__(self, k):
        '''初始化方法

        Parameters
        ----------
        k:int
          邻居的个数
        '''
        self.k = k

    def fit(self, X, y):
        '''训练方法

         Parameters
        ----------
        X: 类数组类型(特征矩阵)。形状为[样本数量,特征数量]
           待训练的样本特征(属性)

        y: 类数组类型(目标标签)。形状为[样本数量]
           每个样本的目标值(标签)
        '''

        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self, X):
        '''
        根据参数传递的X，对样本数据进行预测

         Parameters
        ----------
        X: 类数组类型(特征矩阵)。形状为[样本数量,特征数量]
           待训练的样本特征(属性)

        Returns
        -------
        result: 数组类型
                预测的结果值
        '''
        # 转换成数组类型
        X = np.asarray(X)
        result = []
        for x in X:
            # 计算距离(计算与训练集中每个X的距离)
            dis = np.sqrt(np.sum((x-self.X)**2, axis=1))
            # 返回数组排序后，每个元素在原数组中(排序之前的数组)的索引
            index = dis.argsort()
            # 取前k个距离最近的索引
            index = index[:self.k]
            result.append(np.mean(self.y[index]))
        return result

    def predict2(self, X):
        '''
        根据参数传递的X，对样本数据进行预测(考虑权重)

         Parameters
        ----------
        X: 类数组类型(特征矩阵)。形状为[样本数量,特征数量]
           待训练的样本特征(属性)

        Returns
        -------
        result: 数组类型
                预测的结果值
        '''
        # 转换成数组类型
        X = np.asarray(X)
        result = []
        for x in X:
            # 计算距离(计算与训练集中每个X的距离)
            dis = np.sqrt(np.sum((x-self.X)**2, axis=1))
            # 返回数组排序后，每个元素在原数组中(排序之前的数组)的索引
            index = dis.argsort()
            # 取前k个距离最近的索引
            index = index[:self.k]
            # 求权重
            s = np.sum(1/(dis[index]+0.001))  # 加上0.001，是为了避免距离为0的情况
            # 使用每个节点距离的倒数，除以倒数之和，得到权重
            weight = (1/(dis[index]+0.001))/s
            # 使用邻居节点的标签值，乘以对应的权重，然后相加
            result.append(np.mean(self.y[index]*weight))
        return np.array(result)


t = data.sample(len(data), random_state=0)
train_X = t.iloc[:120, :-1]
train_y = t.iloc[:120, -1]
test_X = t.iloc[120:, :-1]
test_y = t.iloc[120:, -1]

knn = KNN(k=3)
knn.fit(train_X, train_y)
result = knn.predict(test_X)
result2 = knn.predict2(test_X)
print("KNN回归的误差:", np.mean(np.sum((result - test_y)**2)))
print("KNN带权回归的误差:", np.mean(np.sum((result2 - test_y)**2)))

mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(10, 10))
# 绘制预测值
plt.plot(result, "ro-", label="预测值")
plt.plot(result2, "bo-", label="带权预测值")
# 绘制真实值
plt.plot(test_y.values, "go--", label="真实值")
plt.legend()
plt.title("KNN连续纸预测展示")
plt.xlabel("节点序号")
plt.ylabel("花瓣宽度")
plt.show()
