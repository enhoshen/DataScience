import sklearn as sk
import numpy as np
import sys
from sklearn import linear_model , tree , neural_network , svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
def argparse ():
    clstype  =sys.argv[1]
    trainpath=sys.argv[2]
    testpath =sys.argv[3]
    return clstype,trainpath,testpath
class classifier ():
    
    def __init__ (self):
        self.mtype, self.trainfile,self.testfile= argparse()
        self.handIndataparse()
        self.functable={'R':self.Regression,'D':self.DecisionTree,'S':self.Svm,'N':self.NN}
    def handInRun (self):
        self.pred = self.functable[self.mtype]()
        self.genpredcsv(self.pred)
    def handIndataparse (self):
        self.train = np.genfromtxt(self.trainfile,delimiter=',')
        self.trn_x = self.train[:,:-1]
        self.trn_y = self.train[:,-1]
        self.test  = np.genfromtxt(self.testfile ,delimiter=',') 
    def selfCheckRun (self):
        self.pred = self.functable[self.mtype]()
        accArr = [ 1 if x==y else 0 for x,y in zip(self.pred,self.label)]
        accCnt = np.count_nonzero(accArr)
        self.genpredcsv(self.pred)
        print ("accuracy:" + str( accCnt/len(accArr)*100 ))
    def selfCheckdataparse(self):
        self.data = np.genfromtxt( 'spambase.csv' , delimiter=',')
        np.random.shuffle(self.data)
        self.datanum = self.data.shape[0]
        self.train_set_perc = 0.8
        self.datacut = int( self.datanum*self.train_set_perc)
        self.train = self.data[:self.datacut,:]
        self.trn_x = self.train[:,:-1]
        self.trn_y = self.train[:,-1]
        self.test  = self.data[self.datacut:,:-1]
        self.label = self.data[self.datacut:, -1]
    def Regression(self):
        regressor = linear_model.LogisticRegression(
                                    solver = 'liblinear',
                                    multi_class='ovr',
                                    class_weight='balanced')
        regressor.fit(self.trn_x , self.trn_y)
        self.threshold = 0.5 
        return [ 1 if x>self.threshold else 0 for x  in regressor.predict(self.test) ]
    def DecisionTree(self):
        dectree = tree.DecisionTreeClassifier(criterion='entropy')
        dectree.fit(self.trn_x,self.trn_y)
        return dectree.predict(self.test)
    def Svm(self):
        svm = SVC(kernel='linear')
        print ('svc start')
        svm.fit(self.trn_x,self.trn_y)
        print ('svc done')
        return svm.predict(self.test)
    def NN(self):
        scalar = StandardScaler()
        scalar.fit(self.trn_x)
        norm_trn_x = scalar.transform(self.trn_x)
        norm_test_x  = scalar.transform(self.test)

        mlp = neural_network.MLPClassifier(
                        hidden_layer_sizes=(128,),
                        activation='relu',
                        learning_rate='adaptive',
                        solver = 'adam'    
                        )
        mlp.fit( norm_trn_x , self.trn_y)
        return mlp.predict( norm_test_x )


    def genpredcsv( self, predict ):
        with open ( 'predict.csv','w' ) as pfile:
            for row in predict:
                pfile.write( str(int(row))+'\n' )
        return 
if __name__ == '__main__':
    C = classifier()
    C.selfCheckdataparse()
    C.selfCheckRun()    
    C.handInRun()

