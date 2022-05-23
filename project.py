import sys, openpyxl, pandas, sklearn, math, statistics, plotly.graph_objs as go, plotly
from sklearn.cluster import KMeans
from pathlib import Path

#pandas.set_option('display.max_rows', None)
#TODO: same color for clusters and centroids
#TODO: use divergent palette for colorblind people
#TODO: make function for clustering (group by before clustering)
#TODO: make list of subDataframes (see example in spreadsheet)

def calculateAvg(dataset, column):

	for i in range(dataset.shape[0]):
		string = dataset[column].iloc[i][1:-1].replace(',', '')		# [1:-1] removes the first and last char, so '(1.0, 1.0)' becomes '1.0, 1.0'. replace() removes the comma
		minmax = [float(num) for num in string.split()]  			# List with the min and max value of the interval
		dataset[column].iloc[i] = (minmax[0] + minmax[1]) / 2		# Updates the cell with the average


def getKeyFromValue(dictionary, value):
	
	for key, val in dictionary.items():
		if val == value:
			return key
	
	return 0


def getStdDevOfColInCluster(dataset, clusterNumber, column):		# For the 'converted' columns like Fuel, I have to pass kmeans_dataset (and thus add the Cluster column to it, too)

	newList = []

	for i in range(dataset.shape[0]):
		if dataset['ClusterID'].iloc[i] == clusterNumber:
			newList.append(dataset[column].iloc[i])
	
	return statistics.stdev(newList)


def createDict(dataset, column):

	newDict = {}			# key:value --> 2:'C6H6'
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


def findTopClusters(dataset):
	
	clusterDict = createDict(dataset, 'ClusterID')		# Because some clusters have a negative index (OPTICS algorithm)
	numClusters = len(clusterDict)		# Used for other clustering algorithms

	clusterCount = {}	  # key:value --> cluster:numRows, meaning how many rows each cluster has

	for i in range(numClusters):

		count = 0

		for j in range(dataset.shape[0]):

			if dataset['ClusterID'].iloc[j] == i:
				count += 1
		
		clusterCount[i] = count

	# clusterCount is 'sorted' by key/cluster number, so clusterCount[3] gives the number of rows in the third cluster.
	# topClustersDict is 'sorted' based on value, in descending order. So the first key (cluster #) is the one with the most rows

	return sorted(clusterCount.items(), key=lambda x: x[1], reverse=True)


# Returns the clustered dataset and the "cluster" object so it can be used again for predict
def cluster(dataset, fuelsDict, targetDict, expTypeDict, clusterObj, clusteringMode):

	calculateAvg(dataset, 'Phi')
	calculateAvg(dataset, 'Pressure (Bar)')		# Also update the main dataset with the average values for phi, P and T
	calculateAvg(dataset, 'Temperature (K)')	#TODO: add Reactor to clustering parameters(??)

	kmeans_dataset = dataset.drop(columns=['Exp SciExpeM ID', 'Reactor', 'Score', 'Error', 'shift'], axis=1)		# Removes the columns on which I don't want to perform Kmeans

	updateTableFromDict(kmeans_dataset, 'Fuels', fuelsDict)
	updateTableFromDict(kmeans_dataset, 'Target', targetDict)
	updateTableFromDict(kmeans_dataset, 'Experiment Type', expTypeDict)

	if clusteringMode == 'fit_predict':
		dataset['ClusterID'] = clusterObj.fit_predict(kmeans_dataset)
	
	if clusteringMode == 'predict':
		dataset['ClusterID'] = clusterObj.predict(kmeans_dataset)

	return dataset, clusterObj


def plot(topClustersNum, dataset, topClustersDict, clusterDf):			# Prepare 3d scatterplot and then show it
	
	xc = []
	yc = []
	zc = []
	centroidID = []		# The cluster # of the centroid

	for i in range(topClustersNum):

		tempListX = []
		tempListY = []
		tempListZ = []

		for j in range(dataset.shape[0]):
			
			if dataset['ClusterID'].iloc[j] == topClustersDict[i][0]:
				tempListX.append(dataset['Temperature (K)'].iloc[j])		# Used to be kmeans_dataset, but both kmeans_ds and ds/filtered_ds have the same temp/pressure/phi values
				tempListY.append(dataset['Pressure (Bar)'].iloc[j])
				tempListZ.append(dataset['Phi'].iloc[j])
		
		xc.append(sum(tempListX) / topClustersDict[i][1])
		yc.append(sum(tempListY) / topClustersDict[i][1])
		zc.append(sum(tempListZ) / topClustersDict[i][1])
		centroidID.append(topClustersDict[i][0])
	
	centroidDf = pandas.DataFrame(list(zip(xc, yc, zc, centroidID)), columns=['X', 'Y', 'Z', 'ClusterID'])
	
	plt = go.Figure()
	clusterNumbers = list(list(zip(*topClustersDict))[0])		# get list of first element from tuple list. The same numbers of topClustersDict[i][0] below
	plt.add_trace(go.Scatter3d(x=centroidDf['X'], y=centroidDf['Y'], z=centroidDf['Z'], mode='markers', marker=dict(size=5, color=centroidDf['ClusterID'], colorscale='Viridis'), name='Centroid'))

	plt.add_trace(go.Scatter3d(	x=dataset['Temperature (K)'], y=dataset['Pressure (Bar)'], z=dataset['Phi'], customdata=dataset['Fuels'], mode='markers',
									marker=dict(size=2, color=dataset['ClusterID'], colorscale='Viridis'), name=('Cluster '+str(topClustersDict[i][0])),
									hovertemplate='Temperature: %{x} K<br>Pressure: %{y} Bar<br>Phi: %{z}<br>Fuel: %{customdata}'))

	# TODO: put file name in graph title, get from CLI parameter
	plt.update_layout(scene = dict(xaxis_title='Temperature (K)', yaxis_title='Pressure (Bar)', zaxis_title='Phi'), title='Model', legend_title='Legend')
	plt.update_traces()
	plt.show()


def main():

	#firstModelPath = Path(sys.argv[1])		# Get model paths from command line, disabled for now
	#secondModelPath = Path(sys.argv[2])
	kmeans_clusters = 15
	topClustersNum = 5

	clusterObj = sklearn.cluster.KMeans(n_clusters=kmeans_clusters)

	dataset = pandas.read_excel(Path('data', '1800.xlsx'), engine='openpyxl').drop(columns=['Experiment DOI', 'Chem Model', 'Chem Model ID'])
	dataset = dataset.drop(dataset.columns[0], axis=1)		# Removes the first unnamed column

	fuelsDict = createDict(dataset, 'Fuels')  # So I keep the dictionary to analyze the results in each cluster (to reconvert from number to string)
	targetDict = createDict(dataset, 'Target')
	expTypeDict = createDict(dataset, 'Experiment Type')
	reactorDict = createDict(dataset, 'Reactor')

	expTypeDf = dataset.groupby('Experiment Type')
	print(fuelsDict)

	#TODO: check if found sub-dataframes are not empty (is it even needed?)
	subDfList = [] # has dataframes with rows with the same experiment type, reactor and fuel

	for i in range(len(expTypeDict)):		# For each experiment type
		try:
			subDf = expTypeDf.get_group(expTypeDict.get(i))
		except:
			continue		# When I used "pass" I had 375 sub-dataframes, most were duplicates. With continue I only have 50 (can I do it without these ugly try/except??
		reactorSubDf = subDf.groupby('Reactor')

		for j in range(len(reactorDict)):	# For each reactor type
			try:
				fuelsSubDf = reactorSubDf.get_group(reactorDict.get(j))
			except:
				continue
			fuelsSubDf = subDf.groupby('Fuels')

			for k in range(len(fuelsDict)):
				print('current exptype and reactor and fuel: ' + str(expTypeDict.get(i)) + ', ' + str(reactorDict.get(j)) + ', ' + str(fuelsDict.get(k)))
				try:
					finalDf = fuelsSubDf.get_group(fuelsDict.get(k))	# Used to give an error if there are no rows with the current expType and reactor for [CH4, H2]
				except:
					continue
				subDfList.append(finalDf)

	'''for i in range(len(subDfList)):
		print('\n\nPrinting sub-dataframe #' + str(i))
		print(subDfList[i])
'''
	dataset, clusterObj = cluster(dataset, fuelsDict, targetDict, expTypeDict, clusterObj, 'fit_predict')

	print(dataset)

	topClustersDict = findTopClusters(dataset)
	print(topClustersDict)

	
	filteredDataset = keepTopNClusters(dataset, topClustersNum, topClustersDict)

	print(filteredDataset)

	clusterDf = filteredDataset.groupby('ClusterID')
	fuelsDf = filteredDataset.groupby('Fuels')

	#plot(topClustersNum, filteredDataset, topClustersDict, clusterDf)

	for i in range(topClustersNum):
		print('std dev of Score in cluster #' + str(topClustersDict[i][0]) + ': ' + str(clusterDf.get_group(int(topClustersDict[i][0]))['Score'].std()))


	############################################
	########		Second dataset		########
	############################################


	dataset2 = pandas.read_excel(Path('data', '2100_2110.xlsx'), engine='openpyxl').drop(columns=['Experiment DOI', 'Chem Model', 'Chem Model ID'])
	dataset2 = dataset2.drop(dataset2.columns[0], axis=1)		# Removes the first unnamed column
	
	dataset2, clusterObj = cluster(dataset2, fuelsDict, targetDict, expTypeDict, clusterObj, 'predict')

	print(dataset2)

	topClustersDict2 = findTopClusters(dataset2)
	print(topClustersDict2)

	filteredDataset2 = keepTopNClusters(dataset2, topClustersNum, topClustersDict2)

	print(filteredDataset2)

	clusterDf2 = filteredDataset2.groupby('ClusterID')
	fuelsDf2 = filteredDataset2.groupby('Fuels')

	plot(topClustersNum, filteredDataset2, topClustersDict2, clusterDf2)

	print('SECOND MODEL:')

	for i in range(topClustersNum):
		print('std dev of Score in cluster #' + str(topClustersDict2[i][0]) + ': ' + str(clusterDf2.get_group(int(topClustersDict2[i][0]))['Score'].std()))



if __name__ == '__main__':
	main()