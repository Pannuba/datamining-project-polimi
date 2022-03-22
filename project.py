import openpyxl, pandas, sklearn, math, statistics, plotly.express
from sklearn.cluster import kmeans_plusplus, KMeans
from pathlib import Path

#TODO: only use kmeans_dataset instead of mixing dataset and kmeans_dataset
#TODO: parse command line arguments (clustering, fuel type...)

def calculateAvg(dataset, column):

	for i in range(dataset.shape[0]):
		string = dataset[column].iloc[i][1:-1].replace(',', '')	 # [1:-1] removes the first and last char, so "(1.0, 1.0)" becomes "1.0, 1.0". replace() removes the comma
		minmax = [float(num) for num in string.split()]  # List with the min and max value of the interval
		dataset[column].iloc[i] = (minmax[0] + minmax[1]) / 2	 # Updates the cell with the average


def getStdDevOfColInCluster(dataset, clusterNumber, column):		# For the "converted" columns like Fuel, I have to pass kmeans_dataset (and thus add the Cluster column to it, too)

	newList = []

	for i in range(dataset.shape[0]):
		if dataset["ClusterID"].iloc[i] == clusterNumber:
			newList.append(dataset[column].iloc[i])
	
	return statistics.stdev(newList)


def createDictAndUpdateTable(dataset, column):

	newDict = {}			# key:value --> 2:"C6H6"
	j = 0

	for i in range(dataset.shape[0]):
		if dataset[column].iloc[i] not in newDict.values():
			newDict[j] = dataset[column].iloc[i]
			dataset[column].iloc[i] = j
			j += 1
		else:
			dataset[column].iloc[i] = j
	
	return newDict

def filterDataset(dataset, column, value):		# Deletes all rows that don't have a specific value of a column. Used for filtering by fuel type
	
	newDataset = dataset

	for i in range(dataset.shape[0]):
		if dataset[column].iloc[i] != value:
			newDataset = newDataset.drop([i])
			i += 1

	#newDataset = dataset.drop(column, 1)	# Deletes the column to filter by
	
	return newDataset


def main():

	original_dataset = pandas.read_excel(Path('data', '1800.xlsx'), engine='openpyxl')
	dataset = original_dataset.drop(original_dataset.columns[[0,1,3,4]], axis=1)		# Removes the useless columns (TODO: drop by name instead of index)

	#dataset = sklearn.utils.shuffle(dataset)
	print("There are " + str(dataset.shape[0]) + " rows in the dataset")


	# I should do this for each "weird" column... hardcode or find a way to create every dict automatically from the dataset?
	#dataset = filterDataset(dataset, "Fuels", "['C6H6']")
	print("There are " + str(dataset.shape[0]) + " rows in the dataset")

	kmeans_dataset = dataset.drop(dataset.columns[[0,2,8,9,14]], axis=1)		# Removes the columns on which I don't want to perform Kmeans

	fuelsDict = createDictAndUpdateTable(kmeans_dataset, "Fuels")  # So I keep the dictionary to analyze the results in each cluster (to reconvert from number to string)
	targetDict = createDictAndUpdateTable(kmeans_dataset, "Target")
	expTypeDict = createDictAndUpdateTable(kmeans_dataset, "Experiment Type")

	calculateAvg(kmeans_dataset, "Phi")
	calculateAvg(kmeans_dataset, "Pressure (Bar)")
	calculateAvg(kmeans_dataset, "Temperature (K)")

	calculateAvg(dataset, "Phi")
	calculateAvg(dataset, "Pressure (Bar)")		# Also update the main dataset with the average values for phi, P and T
	calculateAvg(dataset, "Temperature (K)")

	print(kmeans_dataset)
	kmeans_clusters = 20		# TODO: pass argument from command line

	cluster = sklearn.cluster.KMeans(n_clusters=kmeans_clusters)
	#cluster = sklearn.cluster.OPTICS()
	#cluster = sklearn.cluster.AffinityPropagation()

	dataset["ClusterID"] = cluster.fit_predict(kmeans_dataset)

	clusterName = []

	for i in range(dataset.shape[0]):
		clusterName.append("C" + str(dataset["ClusterID"].iloc[i]))
	
	dataset["Cluster"] = clusterName

	print(dataset)

	# I have to get d0L1, L2... of the clusters with the most experiments (aka rows) and get their standard deviation.

	clusterDict = createDictAndUpdateTable(dataset, 'ClusterID')		# Because some clusters have a negative index (OPTICS algorithm)
	numClusters = len(clusterDict)
	print("there are " + str(len(clusterDict)) + " clusters")


	clusterCount = {}	  # key:value --> cluster:numRows, meaning how many rows each cluster has

	for i in range(numClusters):

		count = 0

		for j in range(dataset.shape[0]):

			if dataset["ClusterID"].iloc[j] == i:
				count += 1
		
		clusterCount[i] = count

	# clusterCount is "sorted" by key/cluster number, so clusterCount[3][1] gives the number of rows in the third cluster.
	# clusterCountDesc is "sorted" based on value, in descending order. So the first key (cluster #) is the one with the most rows

	clusterCountDesc = sorted(clusterCount.items(), key=lambda x: x[1], reverse=True)
	print(clusterCountDesc)
	biggestCluster = clusterCountDesc[0][0]
	print(str(biggestCluster) + " is the cluster with the most elements")

	for i in range(3):
		print("stdDev(d0L2) in cluster #" + str(clusterCountDesc[i][0]) + " (" + str(clusterCountDesc[i][1]) + " rows) = " + str(getStdDevOfColInCluster(dataset, clusterCountDesc[i][0], 'd0L2')))
		print("stdDev(d1L2) in cluster #" + str(clusterCountDesc[i][0]) + " (" + str(clusterCountDesc[i][1]) + " rows) = " + str(getStdDevOfColInCluster(dataset, clusterCountDesc[i][0], 'd1L2')))
		print("stdDev(d0Pe) in cluster #" + str(clusterCountDesc[i][0]) + " (" + str(clusterCountDesc[i][1]) + " rows) = " + str(getStdDevOfColInCluster(dataset, clusterCountDesc[i][0], 'd0Pe')))		# For cycle, print the standard deviation of the top x clusters?
		print("stdDev(d1Pe) in cluster #" + str(clusterCountDesc[i][0]) + " (" + str(clusterCountDesc[i][1]) + " rows) = " + str(getStdDevOfColInCluster(dataset, clusterCountDesc[i][0], 'd1Pe')) + '\n')

	testDf = dataset.groupby("Fuels")
	print(fuelsDict)
	#print(testDf.get_group(fuelsDict.get(0)))
	'''for i in range(testDf.ngroups):		# For each fuel type
		print("stdDev(d0L2) for" + str(clusterCountDesc[i][0]) + " (" + str(clusterCountDesc[i][1]) + " rows) = " + str(getStdDevOfColInCluster(dataset, clusterCountDesc[i][0], 'd0L2')))
		print("stdDev(d1L2) for" + str(clusterCountDesc[i][0]) + " (" + str(clusterCountDesc[i][1]) + " rows) = " + str(getStdDevOfColInCluster(dataset, clusterCountDesc[i][0], 'd1L2')))
		print("stdDev(d0Pe) for" + str(clusterCountDesc[i][0]) + " (" + str(clusterCountDesc[i][1]) + " rows) = " + str(getStdDevOfColInCluster(dataset, clusterCountDesc[i][0], 'd0Pe')))		# For cycle, print the standard deviation of the top x clusters?
		print("stdDev(d1Pe) for" + str(clusterCountDesc[i][0]) + " (" + str(clusterCountDesc[i][1]) + " rows) = " + str(getStdDevOfColInCluster(dataset, clusterCountDesc[i][0], 'd1Pe')) + '\n')
'''

	# Prepare 3d plot and then show it

	xc = []
	yc = []
	zc = []

	for i in range(numClusters):

		tempListX = []
		tempListY = []
		tempListZ = []

		for j in range(dataset.shape[0]):

			if dataset["ClusterID"].iloc[j] == i:
				tempListX.append(kmeans_dataset["Temperature (K)"].iloc[j])
				tempListY.append(kmeans_dataset["Pressure (Bar)"].iloc[j])
				tempListZ.append(kmeans_dataset["Phi"].iloc[j])
		
		xc.append(sum(tempListX) / clusterCount[i])
		yc.append(sum(tempListY) / clusterCount[i])
		zc.append(sum(tempListZ) / clusterCount[i])

	plt = plotly.express.scatter_3d(dataset, x='Temperature (K)', y='Pressure (Bar)', z='Phi', color="Cluster")
	plt.update_traces(marker={'size':3}) # :3
	plt.show()


if __name__ == '__main__':
	main()