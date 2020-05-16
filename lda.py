from PIL import Image
import numpy as np

def normalize(arr, size, mean):
	mean[0] = np.mean(arr)

	for i in range(size):
		arr[i] -= mean[0]

im = Image.open("Training/s1/1.pgm")
arr = np.array(im)
arr = arr.flatten()
p = 60
subjects = 10

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

cov = np.zeros((p, p), dtype = float)

cov = np.cov(data.transpose())

eig_val, eig_vec = np.linalg.eig(cov)
eig_val = np.absolute(eig_val)
eig_vec = np.absolute(eig_vec)

k = 6
index_list = eig_val.argsort()[-k:][::-1]
feature_vector = np.zeros((p, k), dtype = float)

#print(index_list)

j = 0
for i in index_list:
	feature_vector[:,j] = eig_vec[:,i]
	j += 1

eigen_faces = np.matmul(data, feature_vector)

proj_faces = np.zeros((k, p), dtype = float)
proj_faces = np.matmul(eigen_faces.transpose(), data)

mean_proj = np.zeros(k, dtype = float)
mean_class = np.zeros((k, subjects), dtype = float)

for i in range(k):
	for j in range(p):
		mean_proj[i] += proj_faces[i][j]

	mean_proj[i] = mean_proj[i] / p

for i in range(k):
	for j in range(subjects):
		for lp in range(p / subjects):
			mean_class += proj_faces[i][j * 6 + lp]
	
		mean_class[i][j] = mean_class[i][j] / p


sw = np.zeros((k, k), dtype = float)
sb = np.zeros((k, k), dtype = float)

for i in range(k):
	for j in range(k):
		for l1 in range(subjects):
			for l2 in range(p / subjects):
				sw[i][j] += (proj_faces[i][l1 * 6 + l2] - mean_proj[i]) * (proj_faces[j][l1 * 6 + l2] - mean_proj[j])	
		
			sb += (p / subjects) * (mean_class[i][l1] - mean_proj[i]) * (mean_class[j][l1] - mean_proj[j])

J = np.matmul(np.linalg.inv(sw), sb)

eigval, eigvec = np.linalg.eig(J)
eigval = np.absolute(eigval)
eigvec = np.absolute(eigvec)

m = 6
indexlst = eigval.argsort()[-m:][::-1]
featurevector = np.zeros((k, m), dtype = float)


j = 0
for i in indexlst:
        featurevector[:,j] = eigvec[:,i]
        j += 1
	
fisher_faces = np.matmul(np.transpose(featurevector), proj_faces)


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

		#eigenface_projection = np.zeros((), dtype = float)
		eigenface_projection = np.matmul(eigen_faces.transpose(), arr)

		fisherface_projection = np.matmul(np.transpose(featurevector), eigenface_projection)

		mi = 10000000000000000.0
		min_index = -1
		for i in range(10):
			s = 0.0
			for k in range(6):
				for j in range(fisherface_projection.shape[0]):
					
					#Euclidian Distance
					s += ((fisher_faces[j][i*6 + k] - fisherface_projection[j])  ** 2)
					
					#Manhattan Distance
					#s += abs(fisher_faces[j][i*6 + k] - fisherface_projection[j])
			
			s = s ** 0.5
			#print(s)

			if(s < mi):
				mi = s
				min_index = i

		print("Actual Person: " + str(x1 + 1)  + "Prediction: " + str(min_index + 1))
		
		if(x1 == min_index):
			correct += 1
		else:
			incorrect += 1

print("\nTotal Data: 40, Correct Predictions = " + str(correct) + ", Incorrect Predictions = " + str(incorrect) + ", Percentage =  " + str((correct * 100.0 / 40.0)))
