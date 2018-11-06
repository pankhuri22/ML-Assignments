import cv2
import glob
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from functools import partial
from sklearn.manifold import TSNE

from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# append images ke pixel and respective labels of training datset
train_x = []
train_y = []
for img in glob.glob("/home/pankhuri/Documents/Machine Learning /ML_HW2/Train_val/character_1_ka/*.png"):
	n = cv2.imread(img,0).reshape(-1)
	#print np.shape(n)
	train_x.append(n)
	train_y.append(0)

for img in glob.glob("/home/pankhuri/Documents/Machine Learning /ML_HW2/Train_val/character_2_kha/*.png"):
	n = cv2.imread(img,0).reshape(-1)
	train_x.append(n)
	train_y.append(1)

for img in glob.glob("/home/pankhuri/Documents/Machine Learning /ML_HW2/Train_val/character_3_ga/*.png"):
	n = cv2.imread(img,0).reshape(-1)
	#print np.shape(n)
	train_x.append(n)	
	train_y.append(2)	

for img in glob.glob("/home/pankhuri/Documents/Machine Learning /ML_HW2/Train_val/character_4_gha/*.png"):
	n = cv2.imread(img,0).reshape(-1)
	#print np.shape(n)
	train_x.append(n)
	train_y.append(3)

for img in glob.glob("/home/pankhuri/Documents/Machine Learning /ML_HW2/Train_val/character_5_kna/*.png"):
	n = cv2.imread(img,0).reshape(-1)
	#print np.shape(n)
	train_x.append(n)
	train_y.append(4)		


#normalise dataset
train_x = np.array(train_x)
train_y = np.array(train_y)
mean = np.sum(train_x)/np.size(train_x)
std = np.std(train_x)
train_x = (train_x - mean ) /std

#test dataset
test_x = []
test_y = []


for img in glob.glob("/home/pankhuri/Documents/Machine Learning /ML_HW2/Test/character_1_ka/*.png"):
	n = cv2.imread(img,0).reshape(-1)
	#print np.shape(n)
	test_x.append(n)
	test_y.append(0)


for img in glob.glob("/home/pankhuri/Documents/Machine Learning /ML_HW2/Test/character_2_kha/*.png"):
	n = cv2.imread(img,0).reshape(-1)
	test_x.append(n)
	test_y.append(1)

for img in glob.glob("/home/pankhuri/Documents/Machine Learning /ML_HW2/Test/character_3_ga/*.png"):
	n = cv2.imread(img,0).reshape(-1)
	#print np.shape(n)
	test_x.append(n)	
	test_y.append(2)	

for img in glob.glob("/home/pankhuri/Documents/Machine Learning /ML_HW2/Test/character_4_gha/*.png"):
	n = cv2.imread(img,0).reshape(-1)
	#print np.shape(n)
	test_x.append(n)
	test_y.append(3)

for img in glob.glob("/home/pankhuri/Documents/Machine Learning /ML_HW2/Test/character_5_kna/*.png"):
	n = cv2.imread(img,0).reshape(-1)
	#print np.shape(n)
	test_x.append(n)
	test_y.append(4)	



test_x = np.array(test_x)
test_y = np.array(test_y)

m = np.mean(test_x)
s = np.std(test_x)
test_x = (test_x - m)/s


'''
def grid_search(X_train,y_train):
	Cs = [0.3,0.5,0.1]
	gammas = [0.5, 1]	
	param_grid = {'C': Cs, 'gamma' : gammas}
	nfolds=5
	X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

	clf = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
	clf.fit(X_train,y_train)
	print(clf.best_params_)
'''


grid_search(train_x,train_y)	
# for validation set error
# 0.9405940594059405
#for 21 splits

kf = KFold(n_splits = 21,random_state=None)   # for splits =22 acc=0.9404145077720207
for train_index,test_index in kf.split(train_x):
	X_train, X_test = train_x[train_index] , train_x[test_index]
	Y_train, Y_test = train_y[train_index] , train_y[test_index]

clf = SVC(kernel = 'rbf',gamma=0.001,C=0.1)
clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(Y_test,y_pred)
print accuracy

#for training and test set error

clf = SVC(kernel='rbf',gamma=0.001,C=0.1,probability= True)
clf.fit(train_x,train_y)
y_pred = clf.predict(test_x)

accuracy = accuracy_score(test_y,y_pred)
#print accuracy
#print accuracy
#y_pred = classifier.predict(test_x)
#print accuracy_score(test_y,y_pred)

'''
def plot_decision_surface_sklearn(clf,X,y):
    X0 = X[np.where(y == 0)]
    X1 = X[np.where(y == 1)]
    X2 = X[np.where(y == 2)]

    plt.figure()
    
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
              linestyles=['--', '-', '--'],
              levels=[-.5, 0, .5])
    plt.scatter(X0[:, 0], X0[:, 1], c='r',s=50)
    plt.scatter(X1[:, 0], X1[:, 1], c='b',s=50)
    plt.scatter(X2[:, 0], X2[:, 1], c='y',s=50)

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='None', edgecolors='k')
    plt.show()


X_train_new = TSNE(n_components=2).fit_transform(train_x)
plot_decision_surface_sklearn(clf,X_train_new,train_y)
'''
'''
def confusion_matrix(y_true,y_pred):	
	c= []
	for i in y_pred:
		if i not in c:
			c.append(int(i))
	for i in y_true:
		if i not in c:
			c.append(i)
	cm = np.zeros((len(c),len(c)),dtype=int)
	for i,j in zip(y_pred,y_true):
		cm[int(i)][j] += 1	

	print cm	
	
	plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)
	plt.colorbar()
	plt.tight_layout()
	plt.xlabel('Predicted label')
	plt.ylabel('True label')
	plt.show()

classes = []
for i in train_y :
	if i not in classes:
		classes.append(i)

'''
#confusion_matrix(test_y,y_pred)	
'''
def yo(X_test,y_test,clf,classes):

	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	prob = clf.predict_proba(X_test)[::,1]

	for i in range(len(classes)):
		fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], prob)
    	roc_auc[i] = auc(fpr[i], tpr[i])

	fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), prob.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	all_fpr =np.unique(np.concatenate([fpr[i] for i in range(classes)]))
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(len(classes)):
		mean_tpr += interp(all_fpr,fpr[i],tpr[i])
	mean_tpr /= classes	
	fpr["macro"] = all_fpr
	trp["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

	plt.figure()
	plt.plot(fpr["micro"],tpr["micro"],label = "ROC",color = 'deeppink',linestyle = ':',linewidth=4)
	plt.plot(fpr["macro"],tpr["macro"],label="macro avg ROC",color='navy',linestyle = ':',linewidth=4)
	colors = cycle(['aqua','darkorange','cornflowerblue'])
	for i,color in zip(range(len(classes)),colors) :
		plt.plot(fpr[i],tpr[i],color=color,lw=lw,label="roc for class 0")
	plt.plot([0,1],[0,1],'k--',lw=lw)
	plt.xlim([0.0,1.0])
	plt.ylim([0.0,1.05])
	plt.title('ROC curve')
	plt.legend(loc="lower right")
	plt.show()	
	#parameters = {'C' :(0.1,0.3,0.5)}
	#xx = gridSearch(s,classes,parameters,X_train,y_train)
	#print(s)

#yo(test_x,test_y,clf,classes)	
'''