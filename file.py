import pandas as pd 
import numpy as np 
from keras.preprocessing import image
import matplotlib.pyplot as plt
from pathlib import Path
from mysvm import SVM

plt.style.use('seaborn')

df = pd.read_csv('train.csv')
X = np.array(df.iloc[:,0])
Y = np.array(df.iloc[:,1])
data = []
labels = []

classes = np.unique(Y)
print(classes)

dic =  {}
label = 0
for i in classes:
	dic[i] = label
	label += 1
print(dic)

for i in range(X.shape[0]):
	img = image.load_img(f'./Images/{X[i]}',target_size=(32,32))
	img = image.img_to_array(img)
	data.append(img)
	labels.append(dic[Y[i]])

data = np.array(data,dtype='float32')/255.0 #normalisation (all values between 1s and 0s)
data = data.reshape((data.shape[0],-1))
labels = np.array(labels)
print(data.shape,labels.shape)

def seperate_data(data,labels):
	mapping_dic = {}
	for i in np.unique(labels):
		mapping_dic[i] = []

	for i in range(data.shape[0]):
		mapping_dic[labels[i]].append(data[i])

	for keys in mapping_dic:
		mapping_dic[keys] = np.array(mapping_dic[keys])

	return mapping_dic

mapping_dic = seperate_data(data,labels)

# for keys in mapping_dic:
# 	print(keys,mapping_dic[keys].shape)

def make_Data_table(d1,d2):
	x = np.zeros((d1.shape[0]+d2.shape[0],d1.shape[1]))
	y = np.zeros(d1.shape[0]+d2.shape[0])

	x[:d1.shape[0],:] = d1
	x[d1.shape[0]:,:] = d2

	y[:d1.shape[0]] = +1##################first part is +1
	y[d1.shape[0]:] = -1##################second part is -1

	return x,y


def train(mapping_dic,labels):
	mySVM = SVM()
	parameters = {}

	unique = np.unique(labels)
	for i in range(len(unique)):
		parameters[i] = {}
		for j in range(i+1,len(unique)):
			x,y = make_Data_table(mapping_dic[i],mapping_dic[j])
			weights,bias,loss = mySVM.fit(x,y,itrations=500,learning_rate = 0.00001)
			parameters[i][j] = (weights,bias)

	return parameters

parameters = train(mapping_dic,labels)

def predict(x):

	unique = np.unique(labels)
	count = np.zeros(len(unique))

	for i in range(len(unique)):
		for j in range(i+1,len(unique)):
			weights,bias = parameters[i][j]

			pred = np.dot(weights,x.T)+bias

			if pred>=0:#+1 class
				#therefore prediction is first part(ie i)
				count[i] += 1
			else:#-1 class
				#therefore prediction is second part(ie j)
				count[j] += 1

	return np.argmax(count)


# print(predict(data[27]))
# print(labels[27])

def accuracy(x,y):
	count = 0

	for i in range(x.shape[0]):
		pred = predict(x[i])

		if pred == y[i]:
			count += 1

	print(count/x.shape[0])

accuracy(data,labels)


dftest = pd.read_csv('test.csv')


########-------------------------------Testing predicition on online platform------------------------
dftest = pd.read_csv('test.csv')
# print(dftest.head())
xtest = np.array(dftest.iloc[:,0])
data_test = []

for i in range(xtest.shape[0]):
	img = image.load_img(f'./Images/{xtest[i]}',target_size=(32,32))
	img = image.img_to_array(img)
	data_test.append(img)

data_test = np.array(data_test,dtype='float32')/255.0#normalization
# print(data_test.shape)
data_test = data_test.reshape((data_test.shape[0],-1))
# print(data_test.shape)
test_pred = []

for i in range(data_test.shape[0]):
	ind = predict(data_test[i])
	test_pred.append(classes[ind])

print(test_pred)

test_pred = np.array(test_pred)
# test_pred = np.concatenate((xtest,test_pred),axis=1)
dfpred = pd.DataFrame(test_pred)
dftest['NameOfPokemon'] = dfpred
print(dftest.head())

dftest.to_csv('predOfTest.csv',index=False)






	

