import sys, openpyxl, pandas, sklearn, itertools, math, plotly.graph_objs as go, plotly, numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

#pandas.set_option('display.max_rows', None)

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


def getPermutations(columns, dictList):			# Returns the list of permutations (dict list) given a list of columns. Generic approach to getPermutations

	permutationsList = []
	columnNames = list(columns)
	rows = len(columns[columnNames[0]])

	for i in range(rows):		# For each row
		
		tempDict = {}

		for col in columns:		# For each column (col = key of columns dict = column name). columns = {ColName : [colValues], ...}
		
			tempDict[col] = columns[col][i]	# Add cell values to dict

		if tempDict not in permutationsList:
			permutationsList.append(tempDict)

	return permutationsList


def cluster(dataset, clusterObj, clusteringMode):		# Returns the clustered dataset and the "cluster" object so it can be used again for predict

	kmeans_dataset = dataset.drop(columns=['Exp SciExpeM ID', 'Experiment Type', 'Reactor', 'Target', 'Fuels', 'Error', 'd0L2', 'd1L2', 'd0Pe', 'd1Pe', 'shift'], axis=1)

	calculateAvg(kmeans_dataset, 'Phi')
	calculateAvg(kmeans_dataset, 'Pressure (Bar)')		# Also update the main dataset with the average values for phi, P and T
	calculateAvg(kmeans_dataset, 'Temperature (K)')

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

	plt.add_trace(go.Scatter3d(	x=dataset['Temperature (K)'], y=dataset['Pressure (Bar)'], z=dataset['Phi'], customdata=np.stack((dataset['Fuels'], dataset['ClusterID']), axis=-1), mode='markers',
									marker=dict(size=2, color=dataset['ClusterID'], colorscale='Viridis'), name=('Clusters'),
									hovertemplate='Temperature: %{x} K<br>Pressure: %{y} Bar<br>Phi: %{z}<br>Fuel: %{customdata[0]}<br>Cluster #: %{customdata[1]}'))

	# TODO: put file name in graph title, get from CLI parameter
	plt.update_layout(scene = dict(xaxis_title='Temperature (K)', yaxis_title='Pressure (Bar)', zaxis_title='Phi'), title='Model', legend_title='Legend')
	plt.update_traces()
	plt.show()


def main():

	#firstModelPath = Path(sys.argv[1])		# Get model paths from command line, disabled for now
	#secondModelPath = Path(sys.argv[2])

	pandas.options.mode.chained_assignment = None	# Suppresses the annoying SettingWithCopyWarning
	kmeans_clusters = 5		# Was 15, changed because subclusters are much smaller
	topClustersNum = 2

	clusterObj = sklearn.cluster.KMeans(n_clusters=kmeans_clusters)

	dataset = pandas.read_excel(Path('data', '1800.xlsx'), engine='openpyxl').drop(columns=['Experiment DOI', 'Chem Model', 'Chem Model ID'])
	dataset = dataset.drop(dataset.columns[0], axis=1)		# Removes the first unnamed column

	expTypeDict = createDict(dataset, 'Experiment Type')	# So I keep the dictionary to analyze the results in each cluster (to reconvert from number to string)
	reactorDict = createDict(dataset, 'Reactor')
	#targetDict = createDict(dataset, 'Target')
	fuelsDict = createDict(dataset, 'Fuels')

	dictList = [expTypeDict, reactorDict, fuelsDict]

	columns = {}
	
	columns['Experiment Type'] = dataset['Experiment Type'].tolist()
	columns['Reactor'] = dataset['Reactor'].tolist()
	columns['Fuels'] = dataset['Fuels'].tolist()

	permutations = getPermutations(columns, dictList)	# List of all possible permutations in the dataset (by categoric columns)

	# TODO: calculate median, avg, etc of experiments with a specific experiment type, reactor and fuel. I assume I have to calculate all possible combinations from the dict I found (no user input)

	for i in range(len(permutations)):

		tempDataset = dataset
										# permutations[0] = {'Experiment Type': 'ignition delay measurement', 'Reactor': 'shock tube', 'Fuels': "['H2', 'CO']"}
		for col in permutations[0]:		# Group the dataset for each categoric column. col is the name of the current categoric column
			tempDataset = tempDataset.groupby(col)
			tempDataset = tempDataset.get_group(permutations[i][col])

		if tempDataset.shape[0] > 10:	# Only keep experiment types with more than 10 rows/experiments
			print('\nProcessing experiments with ' + str(permutations[i]))# + ' are:\n' + str(tempDataset) + '\n')
			tempDataset, clusterObj = cluster(tempDataset, clusterObj, 'fit_predict')
			clusterDf = tempDataset.groupby('ClusterID')
			topClustersDict = findTopClusters(tempDataset)

			for j in range(topClustersNum):
				print(	'std dev and median of Score in cluster #' + str(j + 1) + ': ' +
						str(clusterDf.get_group(int(topClustersDict[j][0]))['Score'].std()) + ', ' +
						str(clusterDf.get_group(int(topClustersDict[j][0]))['Score'].median()) )
		
	

if __name__ == '__main__':
	main()