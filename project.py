import openpyxl, pandas, sklearn, seaborn, math, statistics
from sklearn.cluster import kmeans_plusplus, KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

pandas.set_option('display.max_rows', None)

#TODO: only use kmeans_dataset instead of mixing dataset and kmeans_dataset

def calculateAvg(dataset, column):

	for i in range(dataset.shape[0]):
		string = dataset[column].iloc[i][1:-1].replace(',', '')	 # [1:-1] removes the first and last char, so "(1.0, 1.0)" becomes "1.0, 1.0". replace() removes the comma
		minmax = [float(num) for num in string.split()]  # List with the min and max value of the interval
		dataset[column].iloc[i] = (minmax[0] + minmax[1]) / 2	 # Updates the cell with the average


def getStdDevOfColInCluster(dataset, clusterNumber, column):		# For the "converted" columns like Fuel, I have to pass kmeans_dataset (and thus add the Cluster column to it, too)

	newList = []

	for i in range(dataset.shape[0]):
		if dataset["Cluster"].iloc[i] == clusterNumber:
			newList.append(dataset[column].iloc[i])
	
	return statistics.stdev(newList)


def createDictAndUpdateTable(dataset, column):

	newDict = {}			# key:value --> "C6H6":2
	j = 0

	for i in range(dataset.shape[0]):
		if dataset[column].iloc[i] not in newDict:
			newDict[dataset[column].iloc[i]] = j
			j += 1
	
	for i in range(dataset.shape[0]):
		dataset[column].iloc[i] = newDict[dataset[column].iloc[i]]
	
	return newDict

original_dataset = pandas.read_excel(Path('data', '1800.xlsx'), engine='openpyxl')

dataset = original_dataset.drop(original_dataset.columns[[0,1,3,4]], axis=1)		# Removes the useless columns

print("There are " + str(dataset.shape[0]) + " rows in the dataset")


# I should do this for each "weird" column... hardcode or find a way to create every dict automatically from the dataset?

kmeans_dataset = dataset.drop(dataset.columns[[0,2,8,9,14]], axis=1)		# Removes the columns on which I don't want to perform Kmeans


fuelsDict = createDictAndUpdateTable(kmeans_dataset, "Fuels")  # So I keep the dictionary to analyze the results in each cluster (to reconvert from number to string)
targetDict = createDictAndUpdateTable(kmeans_dataset, "Target")
expTypeDict = createDictAndUpdateTable(kmeans_dataset, "Experiment Type")

calculateAvg(kmeans_dataset, "Phi")
calculateAvg(kmeans_dataset, "Pressure (Bar)")
calculateAvg(kmeans_dataset, "Temperature (K)")

print(kmeans_dataset)
kmeans_clusters = 20		# TODO: pass argument from command line

#cluster = sklearn.cluster.KMeans(n_clusters=kmeans_clusters)
cluster = sklearn.cluster.OPTICS()
#cluster = sklearn.cluster.AffinityPropagation()

dataset["Cluster"] = cluster.fit_predict(kmeans_dataset)
print(dataset)

'''
reduced_data = sklearn.decomposition.PCA(n_components=2).fit_transform(kmeans_dataset)
results = pandas.DataFrame(reduced_data,columns=['pca1','pca2'])
seaborn.scatterplot(x="pca1", y="pca2", hue=dataset['Cluster'], data=results)
plt.show()
'''

# I have to get d0L1, L2... of the clusters with the most experiments (aka rows) and get their standard deviation.

clusterDict = createDictAndUpdateTable(dataset, 'Cluster')		# Because some clusters have a negative index (OPTICS algorithm)
numClusters = len(clusterDict)
print("there are " + str(len(clusterDict)) + " clusters")


clusterCount = {}	  # key:value --> cluster:numRows, meaning how many rows each cluster has

for i in range(numClusters):

	count = 0

	for j in range(dataset.shape[0]):

		if dataset["Cluster"].iloc[j] == i:
			count += 1
	
	clusterCount[i] = count

# clusterCount is "sorted" by key/cluster number, so clusterCount[3][1] gives the number of rows in the third cluster.
# clusterCountDesc is "sorted" based on value, in descending order. So the first key (cluster #) is the one with the most rows

print(clusterCount)
clusterCountDesc = sorted(clusterCount.items(), key=lambda x: x[1], reverse=True)
print(clusterCountDesc)
biggestCluster = clusterCountDesc[0][0]
print(str(biggestCluster) + " is the cluster with the most elements")

for i in range(3):
	print("stdDev(d0L2) in cluster #" + str(clusterCountDesc[i][0]) + " (" + str(clusterCountDesc[i][1]) + " rows) = " + str(getStdDevOfColInCluster(dataset, clusterCountDesc[i][0], 'd0L2')))
	print("stdDev(d1L2) in cluster #" + str(clusterCountDesc[i][0]) + " (" + str(clusterCountDesc[i][1]) + " rows) = " + str(getStdDevOfColInCluster(dataset, clusterCountDesc[i][0], 'd1L2')))
	print("stdDev(d0Pe) in cluster #" + str(clusterCountDesc[i][0]) + " (" + str(clusterCountDesc[i][1]) + " rows) = " + str(getStdDevOfColInCluster(dataset, clusterCountDesc[i][0], 'd0Pe')))		# For cycle, print the standard deviation of the top x clusters?
	print("stdDev(d1Pe) in cluster #" + str(clusterCountDesc[i][0]) + " (" + str(clusterCountDesc[i][1]) + " rows) = " + str(getStdDevOfColInCluster(dataset, clusterCountDesc[i][0], 'd1Pe')) + '\n')

# Prepare 3d plot and then show it

x = [float(i) for i in kmeans_dataset['Temperature (K)']]
y = [float(i) for i in kmeans_dataset['Pressure (Bar)']]
z = [float(i) for i in kmeans_dataset['Phi']]

xc = []
yc = []
zc = []

for i in range(numClusters):

	tempListX = []
	tempListY = []
	tempListZ = []

	for j in range(dataset.shape[0]):

		if dataset["Cluster"].iloc[j] == i:
			tempListX.append(kmeans_dataset["Temperature (K)"].iloc[j])
			tempListY.append(kmeans_dataset["Pressure (Bar)"].iloc[j])
			tempListZ.append(kmeans_dataset["Phi"].iloc[j])
	
	xc.append(sum(tempListX) / clusterCount[i])
	yc.append(sum(tempListY) / clusterCount[i])
	zc.append(sum(tempListZ) / clusterCount[i])

#print("xc: " + str(xc))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Pressure (Bar)')
ax.set_zlabel('Phi')
ax.scatter(x, y, z, c=dataset['Cluster'], alpha=0.2, linewidths=0)
ax.scatter(xc, yc, zc, c='red', s=30, alpha=1)
#ax.scatter([10, 20], [10, 20], [10, 20], c='red', s=50)
plt.show()