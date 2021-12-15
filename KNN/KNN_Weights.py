import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

# 读取鸢尾花数据集，header参数来指定标题的行，默认是0，如果没有标题，参数设置为None
data = pd.read_csv(r"C:\\Users\\不归客\Desktop\\iris.csv", header=0)

# 随机抽取一定的数据，默认为1行
# print(data.sample())

# 数据清洗，将Species列转为数值类型
data["Species"] = data["Species"].map(
    {"versicolor": 0, "setosa": 1, "virginica": 2})
# 删除不需要的Id列,默认是删除行，axis=1是指删除列
# drop是删除副本中的数据，inplace=True是指用副本替换掉原本
data.drop("Id", axis=1, inplace=True)
# 判断数据集中是否有重复项,只要有重复项，结果就为True
# data.duplicated().any()
if data.duplicated().any():
    # 删除重复项
    data.drop_duplicates(inplace=True)

# 查看每个类别的鸢尾花有多少条数据
print(data["Species"].value_counts())


class KNN:
    def __init__(self, k):
        '''初始化方法

        Parameters
        --------
        K：邻居的个数
        '''
        self.k = k

    def fit(self, X, y):
        '''训练方法

        Parameters
        --------
        X:类数组类型，形状为[样本数量，特征数量] [149,4]
          待训练的样本特征(属性)
        y:类数组类型，形状为[样本数量]
          每个样本的目标值(标签)
        '''
        # 将X转换为ndarray数组类型
        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def predict(self, X):
        '''根据参数传递的样本，进而对样本数据进行预测(考虑权重，距离越远权重越小，使用距离的倒数作为权重)

        Parameters
        --------
        X:类数组类型，形状为[样本数量，特征数量] [149,4]
          待训练的样本特征(属性)

        Returns
        -------
        result:数组类型
            预测的结果
        '''
        X = np.asarray(X)
        result = []
        # 对ndarray数组进行遍历，每次取数组中的一行
        for x in X:
            # sum默认是将所有的数值求和，axis=1是按行求和
            dis = np.sqrt(np.sum(((x - self.X) ** 2), axis=1))
            # 返回数组排序后每个数组在原数组中的索引
            index = dis.argsort()
            # 进行截断，只取前k个元素
            index = index[:self.k]
            # bincount返回每个元素出现的次数，元素必须是非负的整数[使用weights考虑权重，权重为距离的倒数]
            count = np.bincount(self.y[index], weights=1/dis[index])
            # 返回ndarray数组中，值最大的元素对应的索引，该索引就是我们判定的类别
            # 最大元素，就是出现次数最多的元素
            result.append(count.argmax())
        return np.asarray(result)


# 将数据集随机打乱
t0 = data[data["Species"] == 0]
t1 = data[data["Species"] == 1]
t2 = data[data["Species"] == 2]
# 对每个类别数据进行洗牌
t0 = t0.sample(len(t0), random_state=0)
t1 = t1.sample(len(t1), random_state=0)
t2 = t2.sample(len(t2), random_state=0)
# 构建训练集与测试集
train_X = pd.concat(
    [t0.iloc[:40, :-1], t1.iloc[:40, :-1], t2.iloc[:40, :-1]], axis=0)
train_y = pd.concat(
    [t0.iloc[:40, -1], t1.iloc[:40, -1], t2.iloc[:40, -1]], axis=0)
test_X = pd.concat(
    [t0.iloc[40:, :-1], t1.iloc[40:, :-1], t2.iloc[40:, :-1]], axis=0)
test_y = pd.concat(
    [t0.iloc[40:, -1], t1.iloc[40:, -1], t2.iloc[40:, -1]], axis=0)

# 创建KNN对象，进行训练与测试
knn = KNN(k=3)
# 进行训练
knn.fit(train_X, train_y)
# 进行测试，获得测试结果
result = knn.predict(test_X)
print("预测正确的个数:", np.sum(result == test_y), "测试集的总数:", len(
    test_y), "预测准确率:", np.sum(result == test_y)/len(test_y))


'''
将预测结果可视化展示
'''
# 设置画布大小
plt.figure(figsize=(10, 10))
# 设置参数，保证可以中文显示
mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False
# 训练集数据，绘制散点图，选择其中两个属性进行绘制
plt.scatter(x=t0["Sepal.Length"][:40], y=t0["Petal.Length"]
            [:40], color="r", label='virginica')
plt.scatter(x=t1["Sepal.Length"][:40], y=t1["Petal.Length"]
            [:40], color="green", label='setosa')
plt.scatter(x=t2["Sepal.Length"][:40], y=t2["Petal.Length"]
            [:40], color="b", label='versicolor')
# 绘制测试集数据
right = test_X[result == test_y]
wrong = test_X[result != test_y]
plt.scatter(x=right["Sepal.Length"], y=right["Petal.Length"],
            color='c', marker="x", label='right')
plt.scatter(x=wrong["Sepal.Length"], y=wrong["Petal.Length"],
            color='m', marker=">", label='wrong')
plt.xlabel("花萼长度")
plt.ylabel("花瓣长度")
plt.title("KNN分类结果显示")
plt.legend()
plt.show()
