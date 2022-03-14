import openpyxl, pandas, sklearn, seaborn, math, statistics
from sklearn.cluster import kmeans_plusplus, KMeans
import matplotlib.pyplot as plt
from pathlib import Path

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
numClusters = 30

cluster = sklearn.cluster.KMeans(n_clusters=numClusters)
dataset["Cluster"] = cluster.fit_predict(kmeans_dataset)
print(dataset)

reduced_data = sklearn.decomposition.PCA(n_components=2).fit_transform(kmeans_dataset)
results = pandas.DataFrame(reduced_data,columns=['pca1','pca2'])
seaborn.scatterplot(x="pca1", y="pca2", hue=dataset['Cluster'], data=results)
plt.show()

# I have to get d0L1, L2... of the clusters with the most experiments (aka rows) and get their standard deviation.

clusterCount = {}	  # key:value --> cluster:numRows, meaning how many rows each cluster has

for i in range(numClusters):

	count = 0

	for j in range(dataset.shape[0]):

		if dataset["Cluster"].iloc[j] == i:
			count += 1
	
	clusterCount[i] = count

clusterCount = sorted(clusterCount.items(), key=lambda x: x[1], reverse=True)   # Orders the dictionary based on value, in descending order. So the first key (cluster #) is the one with the most rows
print(clusterCount)
biggestCluster = clusterCount[0][0]
print(str(biggestCluster) + " is the cluster with the most elements")

for i in range(3):
	print("stdDev(d0L2) in cluster #" + str(clusterCount[i][0]) + " (" + str(clusterCount[i][1]) + " rows) = " + str(getStdDevOfColInCluster(dataset, clusterCount[i][0], 'd0L2')))
	print("stdDev(d1L2) in cluster #" + str(clusterCount[i][0]) + " (" + str(clusterCount[i][1]) + " rows) = " + str(getStdDevOfColInCluster(dataset, clusterCount[i][0], 'd1L2')))
	print("stdDev(d0Pe) in cluster #" + str(clusterCount[i][0]) + " (" + str(clusterCount[i][1]) + " rows) = " + str(getStdDevOfColInCluster(dataset, clusterCount[i][0], 'd0Pe')))		# For cycle, print the standard deviation of the top x clusters?
	print("stdDev(d1Pe) in cluster #" + str(clusterCount[i][0]) + " (" + str(clusterCount[i][1]) + " rows) = " + str(getStdDevOfColInCluster(dataset, clusterCount[i][0], 'd1Pe')) + '\n')
