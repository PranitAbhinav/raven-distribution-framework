import ravop.core as R
from ravop.core import Tensor,Scalar
from ravop.utils import inform_server
import numpy as np

def eucledian_distance(self, X, Y):
    return R.square_root(((R.sub(X, Y)).pow(Scalar(2))).sum(axis=0))

'''
            Regressor
'''
class KNN_regressor():

    def __init__(self):
        self.k=None
        self.X=None
        self.y=None
        self.weights=None
        pass

    def fit(self,X_Train,Y_train):
        self.X=Tensor(X_Train)
        self.Y=Tensor(Y_train)
        pass

    def knearestneighbours(self):
        pass


    def predict(self, X_test):
        pass

'''
                    classifier
'''


class KNN_classifier():
    def __init__(self,X_train,Y_train,n_neighbours=None,n_classes=None):
        self.k = n_neighbours
        self.n_c= n_classes
        self.n=len(X_train)
        self.X_train=Tensor(X_train)
        self.Y=Tensor(Y_train)
        pass

    def eucledian_distance(self,X):
        X = R.expand_dims(X, axis=1)
        return R.square_root(R.sub(X, self.X_train).pow(Scalar(2)).sum(axis=2))

    def fit(self, X):
        self.n_q=len(X)
        self.X = Tensor(X)
        d_list=self.eucledian_distance(self.X)
        while d_list.status!="computed":
            pass
        #print(d_list)
        fe=d_list.foreach(operation='sort')
        sl= fe.foreach(operation='slice',begin=0,size=self.k)
        while sl.status != "computed":
            pass
        #print(sl)
        li = sl.output.tolist()
        for i in range(self.n_q):
            row=R.gather(d_list,Tensor([i])).reshape(shape=[self.n])
            while row.status!='computed':
                pass

            #print(row)
            ind=R.find_indices(row,values=li[i])
            while ind.status!='computed':
                pass
            #ind.foreach()
            #print(ind)
            ind=ind.foreach(operation='slice',begin=0,size=1)
            y_neighbours= R.gather(self.Y,ind)
            while y_neighbours.status!='computed':
                pass

            print(y_neighbours)
        pass


#from sklearn.neighbors import KNeighborsClassifier
X_train=[[1,2],[12,21],[1,1],[10,10],[2,1],[13,22],[12,20],[11,10],[10,12]]
Y_train=[1,2,1,0,1,2,2,0,0]

obj=KNN_classifier(X_train,Y_train,n_neighbours=5,n_classes=2)
print(obj.fit([[12,21],[1,2],[1,1],[10,11],[9,3]] ))
