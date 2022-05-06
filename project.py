import openpyxl, pandas, sklearn, math, statistics, plotly.graph_objs as go, plotly
from sklearn.cluster import kmeans_plusplus, KMeans
from pathlib import Path

#pandas.set_option('display.max_rows', None)

#TODO: compare models using the calculated standard deviations
#TODO: parse command line arguments (clustering, fuel type...)

def calculateAvg(dataset, column):

	for i in range(dataset.shape[0]):
		string = dataset[column].iloc[i][1:-1].replace(',', '')	 # [1:-1] removes the first and last char, so "(1.0, 1.0)" becomes "1.0, 1.0". replace() removes the comma
		minmax = [float(num) for num in string.split()]  # List with the min and max value of the interval
		dataset[column].iloc[i] = (minmax[0] + minmax[1]) / 2	 # Updates the cell with the average

def getKeyFromValue(dictionary, value):
	
	for key, val in dictionary.items():
		if val == value:
			return key
	
	return 0

def getStdDevOfColInCluster(dataset, clusterNumber, column):		# For the "converted" columns like Fuel, I have to pass kmeans_dataset (and thus add the Cluster column to it, too)

	newList = []

	for i in range(dataset.shape[0]):
		if dataset["ClusterID"].iloc[i] == clusterNumber:
			newList.append(dataset[column].iloc[i])
	
	return statistics.stdev(newList)

def createDict(dataset, column):

	newDict = {}			# key:value --> 2:"C6H6"
	j = 0

	for i in range(dataset.shape[0]):
		if dataset[column].iloc[i] not in newDict.values():
			newDict[j] = dataset[column].iloc[i]
			j += 1
	
	return newDict

def updateTableFromDict(table, column, dictionary):

	for i in range(table.shape[0]):
		table[column].iloc[i] = getKeyFromValue(dictionary, table[column].iloc[i])

def filterDataset(dataset, column, value):		# Deletes all rows that don't have a specific value of a column. Used for filtering by fuel type
	
	newDataset = dataset

	for i in range(dataset.shape[0]):
		if dataset[column].iloc[i] != value:
			newDataset = newDataset.drop([i])
			i += 1
	
	return newDataset

def keepTopNClusters(dataset, n, topClustersDict):		# topClustersDict: {[#cluster : #rows], ...}, topClustersDict[i][0] gives the cluster number, [i][1] the number of rows for cluster #i
	
	newDataset = dataset
	topClusters = []

	for i in range(n):
		topClusters.append(topClustersDict[i][0])

	for i in range(dataset.shape[0]):
		if dataset['ClusterID'].iloc[i] not in topClusters:
			newDataset = newDataset.drop([i])
	
	return newDataset

def plot(topClustersNum, dataset, topClustersDict, clusterDf):			# Prepare 3d scatterplot and then show it
	
	xc = []
	yc = []
	zc = []

	for i in range(topClustersNum):

		tempListX = []
		tempListY = []
		tempListZ = []

		for j in range(dataset.shape[0]):
			
			if dataset["ClusterID"].iloc[j] == topClustersDict[i][0]:
				tempListX.append(dataset["Temperature (K)"].iloc[j])		# Used to be kmeans_dataset, but both kmeans_ds and ds/filtered_ds have the same temp/pressure/phi values
				tempListY.append(dataset["Pressure (Bar)"].iloc[j])
				tempListZ.append(dataset["Phi"].iloc[j])
		
		xc.append(sum(tempListX) / topClustersDict[i][1])
		yc.append(sum(tempListY) / topClustersDict[i][1])
		zc.append(sum(tempListZ) / topClustersDict[i][1])
	
	plt = go.Figure()
	plt.add_trace(go.Scatter3d(x=xc, y=yc, z=zc, mode='markers', marker=dict(size=5, color='red'), name="Centroid"))

	for i in range(topClustersNum):
		tempDf = clusterDf.get_group(topClustersDict[i][0])
		plt.add_trace(go.Scatter3d(	x=tempDf['Temperature (K)'], y=tempDf['Pressure (Bar)'], z=tempDf['Phi'], customdata=tempDf['Fuels'], mode='markers',
									marker=dict(size=2, color=topClustersDict[i][0]), name=("Cluster "+str(topClustersDict[i][0])),
									hovertemplate='Temperature: %{x} K<br>Pressure: %{y} Bar<br>Phi: %{z}<br>Fuel: %{customdata}'))

	# TODO: put file name in graph title, get from CLI parameter
	plt.update_layout(scene = dict(xaxis_title="Temperature (K)", yaxis_title="Pressure (Bar)", zaxis_title="Phi"), title="Model", legend_title="Legend")
	plt.update_traces()
	plt.show()

def main():

	kmeans_clusters = 15		# TODO: pass argument from command line
	topClustersNum = 5

	original_dataset = pandas.read_excel(Path('data', '1800.xlsx'), engine='openpyxl')
	dataset = original_dataset.drop(original_dataset.columns[[0,1,3,4]], axis=1)		# Removes the useless columns (TODO: drop by name instead of index)

	calculateAvg(dataset, "Phi")
	calculateAvg(dataset, "Pressure (Bar)")		# Also update the main dataset with the average values for phi, P and T
	calculateAvg(dataset, "Temperature (K)")

	kmeans_dataset = dataset.drop(dataset.columns[[0,2,8,9,14]], axis=1)		# Removes the columns on which I don't want to perform Kmeans

	fuelsDict = createDict(kmeans_dataset, "Fuels")  # So I keep the dictionary to analyze the results in each cluster (to reconvert from number to string)
	targetDict = createDict(kmeans_dataset, "Target")
	expTypeDict = createDict(kmeans_dataset, "Experiment Type")

	updateTableFromDict(kmeans_dataset, "Fuels", fuelsDict)
	updateTableFromDict(kmeans_dataset, "Target", targetDict)
	updateTableFromDict(kmeans_dataset, "Experiment Type", expTypeDict)

	cluster = sklearn.cluster.KMeans(n_clusters=kmeans_clusters)
	#cluster = sklearn.cluster.OPTICS()
	#cluster = sklearn.cluster.AffinityPropagation()

	dataset["ClusterID"] = cluster.fit_predict(kmeans_dataset)

	print(dataset)

	# I have to get d0L1, L2... of the clusters with the most experiments (aka rows) and get their standard deviation.

	clusterDict = createDict(dataset, 'ClusterID')		# Because some clusters have a negative index (OPTICS algorithm)
	numClusters = len(clusterDict)
	print("there are " + str(len(clusterDict)) + " clusters")

	clusterCount = {}	  # key:value --> cluster:numRows, meaning how many rows each cluster has

	for i in range(numClusters):

		count = 0

		for j in range(dataset.shape[0]):

			if dataset["ClusterID"].iloc[j] == i:
				count += 1
		
		clusterCount[i] = count

	# clusterCount is "sorted" by key/cluster number, so clusterCount[3] gives the number of rows in the third cluster.
	# topClustersDict is "sorted" based on value, in descending order. So the first key (cluster #) is the one with the most rows

	topClustersDict = sorted(clusterCount.items(), key=lambda x: x[1], reverse=True)
	print(topClustersDict)

	filteredDataset = keepTopNClusters(dataset, topClustersNum, topClustersDict)

	print(filteredDataset)

	clusterDf = filteredDataset.groupby("ClusterID")
	fuelsDf = filteredDataset.groupby("Fuels")

	plot(topClustersNum, filteredDataset, topClustersDict, clusterDf)

	for i in range(topClustersNum):
		print("std dev of Score in cluster #" + str(topClustersDict[i][0]) + ": " + str(clusterDf.get_group(int(topClustersDict[i][0]))['Score'].std()))



if __name__ == '__main__':
	main()