from PIL import Image
import numpy as np

def normalize(arr, size, mean):
	mean[0] = np.mean(arr)

	for i in range(size):
		arr[i] -= mean[0]

def cov_matrix(data, cov):
	print(data.shape)
	print(cov.shape)
	for i in range(cov.shape[0]):
		print(i)
		for j in range(cov.shape[1]):
			s = 0.0
			for k in range(data.shape[0]):
				s = s + data[k][i] * data[k][j]

			cov[i][j] = s / data.shape[0]

	return

im = Image.open("Training/s1/1.pgm")
arr = np.array(im)
arr = arr.flatten()
p = 60

data = np.zeros((arr.shape[0], p), dtype = int)

for i in range(10):
	for j in range(6):
		str_arg = "Training/s" + str(i + 1) + "/" + str(j + 1) + ".pgm"
		im = Image.open(str_arg)
		arr = np.array(im)
		arr = arr.flatten()
		arr = arr.astype(float)
		data[:,i*6 + j] = arr


mean = np.zeros((data.shape[0], 1), dtype = float)

for i in range(data.shape[0]):
	normalize(data[i], p, mean[i])


#print(mean.shape)
cov = np.zeros((p, p), dtype = float)

cov = np.cov(data.transpose())
#print(cov.shape)

eig_val, eig_vec = np.linalg.eig(cov)
eig_val = np.absolute(eig_val)
eig_vec = np.absolute(eig_vec)

#print(eig_val)

k = 10
index_list = eig_val.argsort()[-k:][::-1]
feature_vector = np.zeros((p, k), dtype = float)

#print(index_list)

j = 0
for i in index_list:
	feature_vector[:,j] = eig_vec[:,i]
	j += 1

eigen_faces = np.matmul(data, feature_vector)

signature_faces = np.zeros((k, p), dtype = float)
signature_faces = np.matmul(eigen_faces.transpose(), data)

#print(signature_faces)

correct = 0
incorrect = 0

for x1 in range(10):
	for x2 in range(7,11):

		str_arg = "Training/s" + str(x1 + 1) + "/" + str(x2) + ".pgm"
		im = Image.open(str_arg)
		arr = np.array(im)
		arr = arr.flatten()
		arr = arr.astype(float)


		for i in range(arr.shape[0]):
			arr[i] -= mean[i][0]

		projection = np.matmul(eigen_faces.transpose(), arr)

		mi = 10000000000000000.0
		min_index = -1
		for i in range(10):
			s = 0.0
			for k in range(6):
				for j in range(projection.shape[0]):
					
					#Euclidian Distance
					#s += ((signature_faces[j][i*6 + k] - projection[j])  ** 2)
					
					#Manhattan Distance
					s += abs(signature_faces[j][i*6 + k] - projection[j])
			
			s = s ** 0.5
			#print(s)

			if(s < mi):
				mi = s
				min_index = i

		print("Actual Person: " + str(x1 + 1)  + " Prediction: " + str(min_index + 1))
		
		if(x1 == min_index):
			correct += 1
		else:
			incorrect += 1

print("\nTotal Data: 40, Correct Predictions = " + str(correct) + ", Incorrect Predictions = " + str(incorrect) + ", Percentage =  " + str((correct * 100.0 / 40.0)))
