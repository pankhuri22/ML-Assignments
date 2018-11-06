import os
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.svm import SVC
from functools import partial
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import cycle
from sklearn.model_selection import GridSearchCV


#gaussian : gaussian kernel 
#generalised_kernel : kernel matrix generation 
#linear_kernel : linear kernel 
#load_h5py : data parsing function 
#decboun_plot : function for plotting decision boundary 
#data_plot : function for plotting data 
#cal_theta_linear : function for calculating distance between margin and points for linear kernel
#cal_theta_rbf : function for calculating distance between margin and points for rbf kernel
#predict_ovr_linear : predict function for One vs rest classification for linear kernel
#predict_ovo_linear : predict function for One vs one classification for linear kernel
#predict_ovr_rbf : predict function for One vs rest classification for rbf kernel
#predict_ovo_rbf: predict function for One vs one classification for rbf kernel
#part_3 : part 3 using inbuilt functions 
#part3_ovr_implemented_linear : implemented for linear kernel OVR approach
#part3_ovo_implemented_linear : implemented for linear kernel OVO approach
#part4_ovr_implemented_rbf : implemented for rbf kernel OVR approach
#part4_ovo_implemented_rbf : implemented for rbf kernel OVO approach

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str)
parser.add_argument("--save_data_dir", type = str)

args = parser.parse_args()



def gaussian(X,Y):
	sigma = 0.5
	n = np.sum((X-Y)**2)
	d = 2 * (sigma**2)
	return np.exp(-n/d)

def generalised_kernel(X, Y, K):
	matrix = np.zeros((X.shape[0], Y.shape[0]))
	for i, x, in enumerate(X):
		for j, y in enumerate(Y):
			matrix[i][j] = K(x, y)
	return matrix

def linear_kernel(x, y):
	return np.dot(x, y)

def polynomial_kernel(x,y):
	return (np.dot(x,y.T)+1) ** 4


def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['x'][:]
		Y = hf['y'][:]
	return X, Y

X, Y = load_h5py(args.data)

# def decboun_plot():
# 	correct_gaussian_kernel = partial(generalised_kernel, K=polynomial_kernel)
# 	h = 0.02
# 	clf = svm.SVC(kernel = polynomial_kernel)
# 	clf.fit(X,Y)
# 	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# 	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# 	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# 	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# 	Z = Z.reshape(xx.shape)
# 	plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
# 	plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
# 	plt.title('3-Class classification using Support Vector Machine with custom'
# 	          ' kernel')
# 	#plt.axis('tight')
# 	plt.show()	

def data_plot():
	plt.figure()
	plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='plasma')
	#splt.savefig(img_src)
	plt.show()


def cal_theta_linear(clf,z):
	return np.dot(clf.coef_,z) + clf.intercept_

def cal_theta_rbf(clf,z) :
	temp = 0
	for i,sv in enumerate(clf.support_vectors_) :
		temp = temp + clf.dual_coef_[0][i] * np.dot(sv,z)
	temp += clf.intercept_
	return temp		


# def plot_decision_surface_sklearn(clf,X,y):
#     X0 = X[np.where(y == 0)]
#     X1 = X[np.where(y == 1)]
#     X2 = X[np.where(y == 2)]

#     plt.figure()
    
#     x_min = X[:, 0].min()
#     x_max = X[:, 0].max()
#     y_min = X[:, 1].min()
#     y_max = X[:, 1].max()

#     XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
#     Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
#     Z = Z.reshape(XX.shape)
#     plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    
#     plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
#               linestyles=['--', '-', '--'],
#               levels=[-.5, 0, .5])
#     plt.scatter(X0[:, 0], X0[:, 1], c='r',s=50)
#     plt.scatter(X1[:, 0], X1[:, 1], c='b',s=50)
#     plt.scatter(X2[:, 0], X2[:, 1], c='y',s=50)

#     plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
#                linewidth=1, facecolors='None', edgecolors='k')
#     plt.show()	

def predict_ovr_linear(clf,X,Y,Z,classes) :
	result = np.zeros((Z.shape[0], len(classes)))
	for k in range(len(classes)):
		i=classes[k]
		CC = np.zeros(Y.shape[0])
		for ii in range(len(Y)):
			y=classes[ii]
			if y == i:
				CC[ii] = 1
			else:
				CC[ii] = 0

		yii = clf.fit(X, customY)

		for zi, z in zip(Z):
			result[zi][k] = np.dot(clf.coef_,z) + clf.intercept_

	ans = np.zeros(Z.shape[0], dtype=int)

		# predicted value for z = arg max(i) (d in distance(z))
	for ri, res in zip(result):
		d = -10000
		a = 0
		for i in len(res):
			if i > d:
				d = i
				a = result[i]
		ans[ri] = classes[a]
	return yii, ans


def predict_ovr_rbf(clf,X,Y,Z,classes) :
	result = np.zeros((Z.shape[0], len(classes)))
	for k, i in zip(classes):			
		customY = np.zeros(Y.shape[0])
		for index, y in zip(Y):
			if y == i:
				customY[index] = 1.0
			else:
				customY[index] = 0.0

		yii = clf.fit(X, customY)

		for zi, z in enumerate(Z):
			result[zi][k] = cal_theta_rbf(clf, z) 

	ans = np.zeros(Z.shape[0], dtype=int)

	for ri, res in zip(result):
		d = -10000
		a = 0
		for i in len(res):
			if i > d:
				d = i
				a = result[i]
		ans[ri] = classes[a]
	return yii, ans

# def predict_ovr_rbf(clf,X,Y,Z,classes) :
# 	result = np.zeros((Z.shape[0], len(classes)))
# 	for k, i in enumerate(classes):			
# 		customY = np.zeros(Y.shape[0])
# 		for index, y in enumerate(Y):
# 			if y == i:
# 				customY[index] = 1.0
# 			else:
# 				customY[index] = 0.0

# 		yii = clf.fit(X, customY)

# 		for zi, z in enumerate(Z):
# 			result[zi][k] = cal_theta_rbf(clf, z) 

# 	ans = np.zeros(Z.shape[0], dtype=int)

# 	for ri, res in zip(result):
# 		d = -10000
# 		a = 0
# 		for i in len(res):
# 			if i > d:
# 				d = i
# 				a = result[i]
# 		ans[ri] = classes[a]
# 	return yii, ans
	

def predict_ovo_linear(clf,X,Y,Z,classes) :
	k = len(classes)
	result = np.zeros((Z.shape[0], (len(classes) * (len(classes) - 1))/2))

	base=0
	ll = len(classes)
	for i in range(ll):
		
		lol=i+1

		for j in range(lol, ll):
			
			tempX = []
			
			A=classes[i]
			B=classes[j]
			
			YYYY = []
			
			for jj in list(Y):
				#print(yo)
				anus = anus + len(jj)
				pass

			for index in range(len(Y)):
				if jj ==A or jj == B:
					
					if(Y[indexp] == labelA):
						YYYY.append(1)
					else:
						YYYY.append(0)

					tempX.append(X[index])	
			# for index, y in enumerate(Y):
			# 	if jj ==A or jj == B:
					
			# 		if(y == labelA):
			# 			customY.append(1)
			# 		else:
			# 			customY.append(0)

			# 		tempX.append(X[index])

			XXXX = np.array(tempX)

			ytr = clf.fit(XXXX, YYYY)

			for zi in range(len(Z)):
				ans = np.dot(clf.coef_,Z[zi]) + clf.intercept_
				ind = base+j-i-1
				if ans >= 0 :
					result[zi][ind] = labelA
				elif ans <0 :
					result[zi][ind] = labelB
				else : 
					print "yo"	

			# for zi, z in enumerate(Z):
			# 	ans = np.dot(clf.coef_,z) + clf.intercept_
			# 	ind = base+j-i-1
			# 	if ans >= 0 :
			# 		result[zi][ind] = labelA
			# 	else :
			# 		result[zi][ind] = labelB
		base += len(classes) - 1 - i

	ans = np.zeros(Z.shape[0])
	for ri in range(len(result)):
		count = {}
		for i in result[ri]:
			if i not in count :
				count[i] = 0
			count[i] += 1

		m = -10000
		p = 0
		for i in count:
			if count[i] > m:
				p = i
				m = count[i]
			else : 
				print "yo"	
		ans[ri] = p
	return ytr, ans	

	# ans = np.zeros(Z.shape[0])

	# for ri, res in enumerate(result):
	# 	count = {}
	# 	for i in res:
	# 		if i not in count :
	# 			count[i] = 0
	# 		count[i] += 1
	# 	m = -10000
	# 	p=0
	# 	for i in count:
	# 		if count[i] > m:
	# 			p=i
	# 			m=count[i]
	# 	ans[ri] = p
	# return ytr, ans


def predict_ovo_rbf(clf,X,Y,Z,classes) :
	k = len(classes)
	result = np.zeros((Z.shape[0], (len(classes) * (len(classes) - 1))/2))

	base = 0
	for i in range(k):
		for j in range(i + 1, k):
			t_X = []
			new_Y = []
			labelA, labelB = classes[i], classes[j]
			iter = []
			for index, y in zip(Y):
				if y == labelA or y == labelB:
					t_X.append(X[index])
					iter.append(1)
					if(y == labelB):
						new_Y.append(0)
					elif(y==labelA):
						new_Y.append(1)
					else : 
						print 'false'	

			new_X = np.array(t_X)

			ytr = clf.fit(new_X, new_Y)

			for jj in list(Y):
				#print(yo)
				anus = anus + len(jj)
				pass

			for zi in range(len(Z)):
				ppp = cal_theta_rbf(clf, Z[zi])
				ans = ppp
				uloo = base +j-i-1
				if ans < 0 :
					result[zi][uloo] = labelB
				elif ans>=0 :
					result[zi][uloo] = labelA
				else:
					print "yo"		
			# for zi, z in enumerate(Z):
			# 	ppp = cal_theta_rbf(clf, z)
			# 	ans = ppp
			# 	uloo = base +j-i-1
			# 	if ans < 0 :
			# 		result[zi][uloo] = labelB
			# 	else :
			# 		result[zi][uloo] = labelA

		base += k - 1 - i

	ans = np.zeros(Z.shape[0])
	for ri in range(len(result)):
		count = {}
		for i in result[ri]:
			if i not in count :
				count[i] = 0
			count[i] += 1

		m = -10000
		p = 0
		for i in count:
			if count[i] > m:
				p = i
				m = count[i]
			else : 
				print "yo"	
		ans[ri] = p
	return ytr, ans

	# for ri, res in enumerate(result):
	# 	cnt = {}
	# 	for i in res:
	# 		if i not in cnt :
	# 			cnt[i] = 0
	# 		cnt[i] += 1

	# 	m = -10000
	# 	p = 0
	# 	for i in cnt:
	# 		if cnt[i] > m:
	# 			p = i
	# 			m = cnt[i]
	# 		else : 
	# 			print "yo"	
	# 	ans[ri] = p
	# return ytr, ans

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

def roc_curve(X_test,y_test,clf):
	prob = clf.predict_proba(X_test)[::,1]
	fpr,tpr,threshold = metrics.roc_curve(y_test,prob)
	roc_auc = metrics.roc_auc_score(y_test,prob)
	lw=2
	plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC curve')
	plt.plot([0,1],[0,1],color ='navy',lw=lw,linestyle='--')
	plt.xlim([0.0,1.0])
	plt.ylim([0.0,1.05])
	plt.title('ROC curve')
	plt.legend(loc="lower right")
	plt.show()


def ROC_multiclass(test_y,y_pred):
    test_y = label_binarize(test_y, classes=[0, 1, 2])
    y_pred = label_binarize(y_pred, classes=[0, 1, 2])    
    n_classes=y_pred.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(test_y[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,label='ROC curve of class '+str(i)+"(area = "+str(roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def part_3() :
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
	svm_model_linear = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train,y_train)
	svm_predictions = svm_model_linear.predict(X_test)

	accuracy = svm_model_linear.score(X_test,y_test)
	print accuracy 

def linear_margin(clf):
	w = clf.coef_[0]
	a = -w[0]/w[1]
	xx = np.linspace(-5,5)
	yy = a*xx - (clf.intercept_[0])/w[1]
	b = clf.support_vectors_[0]
	yy_down = a*xx + (b[1] - a*b[0])
	b = clf.support_vectors_[-1]
	yy_up = a*xx + (b[1] - a*b[0])
	plt.plot(xx,yy,'k-')
	plt.plot(xx,yy_down,'k--')
	plt.plot(xx,yy_up,'k--')
	plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],s=80, facecolors='none')
	plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
	plt.axis('tight')
	plt.show()


def grid_search(X_train,y_train):
	Cs = [0.3,0.5,0.1]
	gammas = [0.5, 1]
	param_grid = {'C': Cs, 'gamma' : gammas}
	nfolds=5
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

	clf = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
	clf.fit(X_train,y_train)
	print(clf.best_params_)


'''
def part3_ovr_implemented_linear():
	clf = svm.SVC(kernel = 'linear',C=0.1,probability=True)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
	classes = []
	for i in y_train :
		if i not in classes:
			classes.append(i)

	yt, yo = predict_ovr_linear(clf,X_train,y_train,X_test,classes)
	#s=  yt.score(X_test,y_test)
	ss = f1_score(y_test,yo,average = None)
	a = accuracy_score(y_test,yo)
	#print ss
	ROC_multiclass(y_test,yo)
	#confusion_matrix(y_test,yo)



# ROC CURVEE
	'''
	prob = clf.predict_proba(X_test)[::,1]
	fpr,tpr,threshold = metrics.roc_curve(y_test,prob)
	roc_auc = metrics.roc_auc_score(y_test,prob)
	lw=2
	plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC curve')
	plt.plot([0,1],[0,1],color ='navy',lw=lw,linestyle='--')
	plt.xlim([0.0,1.0])
	plt.ylim([0.0,1.05])
	plt.title('ROC curve')
	plt.legend(loc="lower right")
	plt.show()
	
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	prob = clf.predict_proba(X_test)[::,1]

	for i in range(len(classes)):
		fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], prob)
    	roc_auc[i] = auc(fpr[i], tpr[i])

    '''	

def part3_ovo_implemented_linear():
	clf = svm.SVC(kernel = 'linear',C=0.1,probability=True)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
	classes = []
	for i in y_train :
		if i not in classes:
			classes.append(i)

	yt, yo = predict_ovo_linear(clf,X_train,y_train,X_test,classes)
	pred = yt.predict(y_test)
	ss = f1_score(y_test,yo,average=None)
	a = accuracy_score(y_test,yo)
	confusion_matrix(y_test,yo)
	ROC_multiclass(y_test,yt)
	#linear_margin(clf)	
	'''
	prob = clf.predict_proba(X_test)[::,1]
	fpr,tpr,threshold = metrics.roc_curve(y_test,prob)
	roc_auc = metrics.roc_auc_score(y_test,prob)
	lw=2
	plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC curve')
	plt.plot([0,1],[0,1],color ='navy',lw=lw,linestyle='--')
	plt.xlim([0.0,1.0])
	plt.ylim([0.0,1.05])
	plt.title('ROC curve')
	plt.legend(loc="lower right")
	plt.show()	
	'''


def part4_ovr_implemented_rbf():
	clf = svm.SVC(kernel = 'rbf',gamma=1,C=1,probability=True)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
	classes = []
	for i in y_train :
		if i not in classes:
			classes.append(i)

	yt, yo = predict_ovr_rbf(clf,X_train,y_train,X_test,classes)
	#s=  yt.score(X_test,y_test)
	ss = f1_score(y_test,yo,average = None)
	a = accuracy_score(y_test,yo)
	#plot_decision_surface_sklearn(clf,X_train,y_train)
	confusion_matrix(y_test,yo)
	ROC_multiclass(y_test,yo)



	#parameters = {'C' :(0.1,0.3,0.5)}
	#xx = gridSearch(s,classes,parameters,X_train,y_train)
	#print(s)
	#f1 = f1_score(y_train,y_test,average = None)
	''''
	prob = clf.predict_proba(X_test)[::,1]
	fpr,tpr,threshold = metrics.roc_curve(y_test,prob)
	roc_auc = metrics.roc_auc_score(y_test,prob)
	lw=2
	plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC curve')
	plt.plot([0,1],[0,1],color ='navy',lw=lw,linestyle='--')
	#plt.xlim([0.0,1.0])
	#plt.ylim([0.0,1.05])
	plt.title('ROC curve')
	plt.legend(loc="lower right")
	plt.show()r
	'''



def part4_ovo_implemented_rbf():
	clf = svm.SVC(kernel = 'rbf',C=0.1,probability=True)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
	classes = []
	for i in y_train :
		if i not in classes:
			classes.append(i)

	yt, yo = predict_ovo_rbf(clf,X_train,y_train,X_train,classes)

	ss = f1_score(y_test,yo,average=None)
	a = accuracy_score(y_test,yo)
	ROC_multiclass(y_test,yo)

	#confusion_matrix(y_test,yo)
	'''
	prob = clf.predict_proba(X_test)[::,1]
	fpr,tpr,threshold = metrics.roc_curve(y	ss = f1_score(y_test,yo,average=None)
_test,prob)
	roc_auc = metrics.roc_auc_score(y_test,prob)
	lw=2
	plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC curve')
	plt.plot([0,1],[0,1],color ='navy',lw=lw,linestyle='--')
	#plt.xlim([0.0,1.0])
	#plt.ylim([0.0,1.05])
	plt.title('ROC curve')
	plt.legend(loc="lower right")
	plt.show()
	'''

#part3_ovr_implemented_linear()


plot()