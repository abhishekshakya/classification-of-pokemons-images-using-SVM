import numpy as np 

class  SVM:
	def __init__(self,c=1.0):
		self.w = 0#lets assume weights as 1-D vector
		self.b = 0#its one of the weights that we have kept seperated
		self.c = c

	def hinge_loss(self,w,b,X,Y):
		loss = 0.5*np.dot(w,w.T)	 	
		for i in range(X.shape[0]):
			ti = Y[i]*(np.dot(X[i],w.T)+b)#scaler
			loss += self.c*(max(0,1-ti))

		return loss


	def fit(self,X,Y,batch_size=100,itrations=300,learning_rate = 0.001):
		'''batch_size for mini batch grad descent and itrations for no of itrations we have to do gradient descent'''
		w = np.zeros(X.shape[1])
		b = 0
		losses = []

		for itr in range(itrations):
			# print(w,b)
			loss = self.hinge_loss(w,b,X,Y)
			losses.append(loss)

			for batch_x in range(0,X.shape[0],batch_size):

				gradient_w = 0
				gradient_b = 0

				for i in range(batch_x,batch_x+batch_size):
					if i<X.shape[0]:
						ti = Y[i]*(np.dot(X[i],w.T)+b)

						if ti>1:
							gradient_w += 0
							gradient_b += 0
						else:
							gradient_w += learning_rate*self.c*Y[i]*X[i]#vector of size X.shape[1] ie features size
							gradient_b += learning_rate*self.c*Y[i]#scaler

				w = w - learning_rate*w + gradient_w
				b = b + gradient_b

		self.w = w
		self.b = b

		return w,b,losses