import pandas as pd,os
import numpy as np
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.neighbors import KNeighborsClassifier

data_path="F:\\datasets\\imageclef2020\\"
features_path=data_path+"features\\"

files=os.listdir(features_path)
train_haralick=[f for f in files if ("train" in f and "haralick" in f)]

for f in train_haralick:
    print(f,pd.read_csv(features_path+f,header=None).shape)

test_haralick=[f.replace("train","test") for f in train_haralick]

train_lbp=[f for f in files if ("train" in f and "lbp" in f)]
test_lbp=[f.replace("train","test") for f in train_lbp]

data = pd.DataFrame()
for i,f in enumerate(train_lbp):
    if(i==0):
        data=pd.read_csv(features_path+f)
    else:
        data=pd.concat([data.iloc[:,:-1], pd.read_csv(features_path+f)], axis=1).reindex(data.index)
data.drop('Unnamed: 0',axis=1,inplace=True)

Test = pd.DataFrame()
for i,f in enumerate(test_lbp):
    if(i==0):
        Test=pd.read_csv(features_path+f)
    else:
        Test=pd.concat([Test.iloc[:,:-1], pd.read_csv(features_path+f)], axis=1).reindex(Test.index)
Test.drop('Unnamed: 0',axis=1,inplace=True)

results_path=data_path+"results\\7363098a-edf2-4d58-a87d-f35564ec7ce0_TrainingSet.csv"
results=pd.read_csv(results_path)
results['Filename'] = results['Filename'].str[8:11]
results['Filename'] = results['Filename'].astype(int)

Train = pd.merge(data, results, how='inner', left_on=data.columns[-1], right_on='Filename')

#Test=pd.read_csv(features_path+f2,header=None)
X_test=Test.iloc[:,:-1]

print(Train.columns)
X=Train.iloc[:,:-7]
Ys=Train.iloc[:,-7:]

from random import randint
test_col=[randint(0,282) for i in range(80)]

results=[pd.DataFrame() for i in range(6)]
f1=np.zeros((6,5),float)
labels=pd.DataFrame()

clf=list()
clf_titles=["LR","KNN","DT","ET","RF","SGD"]
#Logistic Regression
clf.append(LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial'))
#KNN classifier
clf.append(KNeighborsClassifier(n_neighbors=11))
#DecisionTreeClassifier
clf.append(DecisionTreeClassifier(random_state=0))
#ExtraTreeClassifier
clf.append(ExtraTreeClassifier(criterion='entropy'))
#RandomForestClassifier
clf.append(RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0))
#SGDClassifier
clf.append(SGDClassifier(loss="hinge", penalty="l2", n_jobs=7))

for cl,c in enumerate(clf):
    X_test1 = X[X[X.columns[-1]].isin(test_col)]
    X_train = X[~X[X.columns[-1]].isin(test_col)]
    Ys_test1 = Ys[Ys["Filename"].isin(test_col)]
    Ys_train1 = Ys[~Ys["Filename"].isin(test_col)]
    Ys_train=pd.DataFrame()
    Ys_train["Filename"]=(Ys_train1.iloc[:,0])
    Ys_train["Left"]=(Ys_train1.iloc[:,1]*100)+(Ys_train1.iloc[:,3]*10)+Ys_train1.iloc[:,5]
    Ys_train["Right"]=(Ys_train1.iloc[:,2]*100)+(Ys_train1.iloc[:,4]*10)+Ys_train1.iloc[:,6]
    
    Ys_test=pd.DataFrame()
    Ys_test["Filename"]=(Ys_test1.iloc[:,0])
    Ys_test["Left"]=(Ys_test1.iloc[:,1]*100)+(Ys_test1.iloc[:,3]*10)+Ys_test1.iloc[:,5]
    Ys_test["Right"]=(Ys_test1.iloc[:,2]*100)+(Ys_test1.iloc[:,4]*10)+Ys_test1.iloc[:,6]
    
    Ys_test["LungAffected_"+"left"]=[1 if (i==111 or i==110 or i==101 or i==100) else 0 for i in Ys_test["Left"]]
    Ys_test["Caverns_"+"left"]=[1 if (i==111 or i==110 or i==11 or i==10) else 0 for i in Ys_test["Left"]]
    Ys_test["Pleurisy_"+"left"]=[1 if (i==111 or i==101 or i==11 or i==1) else 0 for i in Ys_test["Left"]]
    Ys_test["LungAffected_"+"Right"]=[1 if (i==111 or i==110 or i==101 or i==100) else 0 for i in Ys_test["Right"]]
    Ys_test["Caverns_"+"Right"]=[1 if (i==111 or i==110 or i==11 or i==10) else 0 for i in Ys_test["Right"]]
    Ys_test["Pleurisy_"+"Right"]=[1 if (i==111 or i==101 or i==11 or i==1) else 0 for i in Ys_test["Right"]]
    
    Ys_test.drop("Left",axis=1,inplace=True)
    Ys_test.drop("Right",axis=1,inplace=True)
    
    X_train=X_train.iloc[:,:-1]
    X_test=X_test1.iloc[:,:-1]
    
    for i in range(0,2):
        Y_train=Ys_train.iloc[0:,i+1]
        results[i]=pd.DataFrame(X_test1)
        c.fit(X_train,Y_train)
        Y_pred=c.predict(X_test)
        results[i]=results[i].assign(Predicted=Y_pred)
        results[i]=results[i].iloc[:,-2:]
        df=results[i]
        df['Predicted']=df['Predicted'].astype(object)
        #111,110,101,100,11,10,1
        df["LungAffected_"+"right" if i%2 else "LungAffected_"+"left"]=[1 if (i==111 or i==110 or i==101 or i==100) else 0 for i in df["Predicted"]]
        df["Caverns_"+"right" if i%2 else "Caverns_"+"left"]=[1 if (i==111 or i==110 or i==11 or i==10) else 0 for i in df["Predicted"]]
        df["Pleurisy_"+"right" if i%2 else "Pleurisy_"+"left"]=[1 if (i==111 or i==101 or i==11 or i==1) else 0 for i in df["Predicted"]]
        #sum([1 for i in df["Pleurisy_left"] if i==1])
        meanv=df.groupby([df.columns[0]]).mean()
        labels[meanv.iloc[:,0].name]=meanv.iloc[:,0]
        labels[meanv.iloc[:,1].name]=meanv.iloc[:,1]
        labels[meanv.iloc[:,2].name]=meanv.iloc[:,2]
    Yg=Ys_test.groupby(['Filename']).mean()    
    temp=labels.merge(Yg, how="inner" ,left_index=True,right_index=False,right_on="Filename")
    temp.to_csv(data_path+"results\\results_lbp_combined_"+clf_titles[cl]+".csv")