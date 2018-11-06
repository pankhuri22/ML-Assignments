import os
import argparse
import h5py
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from functools import partial
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split


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



def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['x'][:]
		Y = hf['y'][:]
	return X, Y

X, Y = load_h5py(args.data)

def decboun_plot():
	correct_gaussian_kernel = partial(generalised_kernel, K=gaussian)
	h = 0.02
	clf = svm.SVC(kernel = correct_gaussian_kernel)
	clf.fit(X,Y)
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
	plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
	plt.title('3-Class classification using Support Vector Machine with custom'
	          ' kernel')
	plt.axis('tight')
	plt.show()	

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

def predict_ovr_linear(clf,X,Y,Z,classes) :
	result = np.zeros((Z.shape[0], len(classes)))
	for k, i in enumerate(classes):			
		customY = np.zeros(Y.shape[0])
		for index, y in enumerate(Y):
			if y == i:
				customY[index] = 1.0
			else:
				customY[index] = 0.0

		yii = clf.fit(X, customY)

			# predict the label for unseen sample z
		for zi, z in enumerate(Z):
			result[zi][k] = cal_theta_linear(clf, z) 

	ans = np.zeros(Z.shape[0], dtype=int)

		# predicted value for z = arg max(i) (d in distance(z))
	for ri, res in enumerate(result):
		d = -1e14
		a = 0
		for index, i in enumerate(res):
			if i > d:
				d = i
				a = index
		ans[ri] = classes[a]
	return yii, ans


def predict_ovr_rbf(clf,X,Y,Z,classes) :
	result = np.zeros((Z.shape[0], len(classes)))
	for k, i in enumerate(classes):			
		customY = np.zeros(Y.shape[0])
		for index, y in enumerate(Y):
			if y == i:
				customY[index] = 1.0
			else:
				customY[index] = 0.0

		yii = clf.fit(X, customY)

			# predict the label for unseen sample z
		for zi, z in enumerate(Z):
			result[zi][k] = cal_theta_rbf(clf, z) 

	ans = np.zeros(Z.shape[0], dtype=int)

		# predicted value for z = arg max(i) (d in distance(z))
	for ri, res in enumerate(result):
		d = -1e14
		a = 0
		for index, i in enumerate(res):
			if i > d:
				d = i
				a = index
		ans[ri] = classes[a]
	return yii, ans	
	

def predict_ovo_linear(clf,X,Y,Z,classes) :
	k = len(classes)
	result = np.zeros((Z.shape[0], (len(classes) * (len(classes) - 1))/2))

	base = 0
	for i in range(len(classes)):
		for j in range(i + 1, len(classes)):
			tempX = []
			labelA, labelB = classes[i], classes[j]
			customY = []
			for index, y in enumerate(Y):
				if y == labelA or y == labelB:
					tempX.append(X[index])
					if(y == labelA):
						customY.append(1)
					else:
						customY.append(0)

			customX = np.array(tempX)

			ytr = clf.fit(customX, customY)

			for zi, z in enumerate(Z):
				ans = cal_theta_linear(clf, z)
				if ans >= 0 :
					result[zi][base + j - i - 1] = labelA
				else :
					result[zi][base + j - i - 1] = labelB
		base += len(classes) - 1 - i

	ans = np.zeros(Z.shape[0])

	for ri, res in enumerate(result):
		count = {}
		for i in res:
			if i not in count :
				count[i] = 0
			count[i] += 1

		m = -1
		p = -1
		for i in count:
			if count[i] > m:
				p = i
				m = count[i]
		ans[ri] = p
	return ytr, ans


def predict_ovo_rbf(clf,X,Y,Z,classes) :
	k = len(classes)
	result = np.zeros((Z.shape[0], (k * (k - 1))/2))

	base = 0
	for i in range(k):
		for j in range(i + 1, k):
			tempX = []
			customY = []
			labelA, labelB = classes[i], classes[j]
			for index, y in enumerate(Y):
				if y == labelA or y == labelB:
					tempX.append(X[index])
					if(y == labelA):
						customY.append(1)
					else:
						customY.append(0)

			customX = np.array(tempX)

			ytr = clf.fit(customX, customY)

			for zi, z in enumerate(Z):
				ans = cal_theta_rbf(clf, z)
				if ans >= 0 :
					result[zi][base + j - i - 1] = labelA
				else :
					result[zi][base + j - i - 1] = labelB
		base += k - 1 - i

	ans = np.zeros(Z.shape[0])

	for ri, res in enumerate(result):
		cnt = {}
		for i in res:
			if i not in cnt :
				cnt[i] = 0
			cnt[i] += 1

		m = -1
		p = -1
		for i in cnt:
			if cnt[i] > m:
				p = i
				m = cnt[i]
		ans[ri] = p
	return ytr, ans

'''
def gridSearch(s,classes, parameters, X, Y):
	max_score = 0.0
	best_params = {}
	for c in parameters['C']:
		score = s
		if score > max_score:
			max_score = score
			best_params['C'] = c

	return best_params
'''
def part_3() :
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
	svm_model_linear = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train,y_train)
	svm_predictions = svm_model_linear.predict(X_test)

	accuracy = svm_model_linear.score(X_test,y_test)
	print accuracy 


def part3_ovr_implemented_linear():
	clf = svm.SVC(kernel = 'linear',C=0.1)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
	classes = []
	for i in y_train :
		if i not in classes:
			classes.append(i)

	yt, yo = predict_ovr_linear(clf,X_train,y_train,X_test,classes)
	#s=  yt.score(X_test,y_test)
	ss = f1_score(y_test,yo,average = None)
	a = accuracy_score(y_test,yo)
	#parameters = {'C' :(0.1,0.3,0.5)}
	#xx = gridSearch(s,classes,parameters,X_train,y_train)
	#print(s)
	#f1 = f1_score(y_train,y_test,average = None)
	print ss.shape

def part3_ovo_implemented_linear():
	clf = svm.SVC(kernel = 'linear',C=0.1)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
	classes = []
	for i in y_train :
		if i not in classes:
			classes.append(i)

	yt, yo = predict_ovo_linear(clf,X_train,y_train,X_test,classes)

	ss = f1_score(y_test,yo,average=None)
	a = accuracy_score(y_test,yo)
	print ss, a


def part4_ovr_implemented_rbf():
	clf = svm.SVC(kernel = 'rbf',C=0.1)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
	classes = []
	for i in y_train :
		if i not in classes:
			classes.append(i)

	yt, yo = predict_ovr_rbf(clf,X_train,y_train,X_test,classes)
	#s=  yt.score(X_test,y_test)
	ss = f1_score(y_test,yo,average = None)
	a = accuracy_score(y_test,yo)
	#parameters = {'C' :(0.1,0.3,0.5)}
	#xx = gridSearch(s,classes,parameters,X_train,y_train)
	#print(s)
	#f1 = f1_score(y_train,y_test,average = None)
	print ss

def part4_ovo_implemented_rbf():
	clf = svm.SVC(kernel = 'rbf',C=0.1)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
	classes = []
	for i in y_train :
		if i not in classes:
			classes.append(i)

	yt, yo = predict_ovo_rbf(clf,X_train,y_train,X_test,classes)

	ss = f1_score(y_test,yo,average=None)
	a = accuracy_score(y_test,yo)
	print ss, a



part3_ovo_implemented_rbf()

































































































'''

with h5py.File("data_1.h5") as dataH5:
  # get the keys
  key_list = dataH5.keys()
  for key in key_list:
    # show every matrix for a given key
    matrix = dataH5.get(key)
    #np.reshape(matrix,(matrix.shape[0],matrix.shape[1]))
    #print(type(matrix))
    #break
    plt.plot(matrix,'o')

    #plt.scatter(X,Y)
    plt.show()
    #plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='plasma')
    #plt.plot(matrix)
    #print(matrix)
    #plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    #plt.show()
'''
