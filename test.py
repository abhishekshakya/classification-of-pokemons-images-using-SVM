import pandas as pd 
import numpy as np 
from keras.preprocessing import image
import matplotlib.pyplot as plt
from pathlib import Path
plt.style.use('seaborn')

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

