import numpy as np

class PerceptionLearning(object):
    #这个类初始化的时候应该要哪些属性呢？传入的参数肯定是xtrain,ytrain.那运算中涉及到的其他参数就应该在类的初始化时候定义，比如说
    # 学习到的w的矩阵,阈值threshold，比如说迭代次数
    def __init__(self,eta = 0.1,iters = 10000):
        self.eta = eta #修正w和threshold时的步幅
        self.w = self.threshold= None
        self.iters = iters
        self.error_history = []
    
    def fit(self,xtrain,ytrain):
        self.w = np.zeros(xtrain.shape[1]) #w的长度应该和x的列相等，即特征的个数
        self.threshold = 0
        for _ in range(self.iters):
            error_count = 0 
            for xi,yi in zip(xtrain,ytrain):
            # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
            # 如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
                if yi*self.predict(xi) <= 0 : #yi和预测的yi正负号不一样，即值不一样，预测出错更新w
                    error_count +=1
                    self.w += self.eta*yi*xi #self.w = self.w + xi*yi
                    self.threshold += self.eta*yi #阈值应该这么更新吗？
            self.error_history.append(error_count)
            if error_count ==0:  #没有分类错误的数据的时候就停止迭代
                break
    def predict(self,x):
        ypred = np.dot(x,self.w) + self.threshold
        return np.sign(ypred)

model = PerceptionLearning()
xtrain = np.array([[1,0],[2,0],[2,1],[1,3],[1,2],[2,3]])
ytrain = np.array([1,1,1,-1,-1,-1])
model.fit(xtrain,ytrain)
xtest = [3,4]
ypred = model.predict(xtest)
print(ypred)
print()



     


