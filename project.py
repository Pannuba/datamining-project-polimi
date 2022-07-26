import openpyxl, sklearn, itertools, scipy.stats
from sklearn.cluster import KMeans
from pandas.api.types import is_numeric_dtype
from pathlib import Path
from plotting import *
from dataset import *

#pandas.set_option('display.max_rows', None)

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


def updateTableFromDict(table, column, dictionary):		# Currently unused, clustering is only performed on numerical variables

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


def cluster(dataset, clusterObj, clusteringMode):		# Returns the clustered dataset and the "cluster" object so it can be used again for predict

	kmeans_dataset = dataset.drop(columns=['Experiment Type', 'Reactor', 'Target', 'Fuels', 'Error', 'd0L2', 'd1L2', 'd0Pe', 'd1Pe', 'shift'], axis=1)

	if clusteringMode == 'fit_predict':
		dataset['ClusterID'] = clusterObj.fit_predict(kmeans_dataset)
	
	if clusteringMode == 'predict':
		dataset['ClusterID'] = clusterObj.predict(kmeans_dataset)

	return dataset, clusterObj

def getFinalDf(dataset, permutations, clusterObj):

	finalDf = pandas.DataFrame(columns=['Experiment Type', 'Reactor', 'Target', 'Fuels', 'avg', 'median', 'std', '#'])#, 'cl'])		# Dataframe later used for plotting the bar graph

	for i in range(len(permutations)):

		newRow = []
		tempDataset = dataset.df
										# permutations[0] = {'Experiment Type': 'ignition delay measurement', 'Reactor': 'shock tube', 'Fuels': "['H2', 'CO']"}
		for col in permutations[0]:		# Group the dataset for each categoric column. col is the name of the current categoric column
			tempDataset = tempDataset.groupby(col)
			tempDataset = tempDataset.get_group(permutations[i][col])
			newRow.append(permutations[i][col])		# Start building the new row by adding the permutation's values. If less than 10 rows it resets at the start of the outer for loop

		if tempDataset.shape[0] >= 20:	# Only cluster experiment types with more than 20 rows/experiments
			#print('\nProcessing experiments with ' + str(permutations[i]))# + ' are:\n' + str(tempDataset) + '\n')
			tempDataset, clusterObj = cluster(tempDataset, clusterObj, 'fit_predict')
			clusterDf = tempDataset.groupby('ClusterID')
			topClustersDict = findTopClusters(tempDataset)
			newRow.append(clusterDf.get_group(int(topClustersDict[0][0]))['Score'].mean())	# Only uses the biggest cluster. [0][0] gets the ID, [0][1] gets the amount of rows
			newRow.append(clusterDf.get_group(int(topClustersDict[0][0]))['Score'].median())
			newRow.append(clusterDf.get_group(int(topClustersDict[0][0]))['Score'].std())
			#newRow.append(int(topClustersDict[0][1]))
		
		elif tempDataset.shape[0] >= 10:		# Discards experiment types with less than 10 rows (there would be >>100 total types)
			newRow.append(tempDataset['Score'].mean())
			newRow.append(tempDataset['Score'].median())
			newRow.append(tempDataset['Score'].std())
		
		else:
			continue		# Skips the last two lines in the for loop if it's an experiment type with too few rows

		newRow.append(tempDataset.shape[0])
		finalDf.loc[len(finalDf.index)] = newRow
	
	return finalDf.sort_values(by=['avg'], ascending=False)


def getResultsDf(df1, df2, commonPermutations):
	
	resultsDf = pandas.DataFrame(columns=['Experiment Type', 'Reactor', 'Target', 'Fuels', 'Score'])

	for i in range(len(commonPermutations)):	# Build the dataframe by adding a row for each permutation

		newRow = []
		tempDf1 = df1
		tempDf2 = df2
		
		#print('current permutation is ' + str(commonPermutations[i]['Experiment Type']) + ', ' + str(commonPermutations[i]['Reactor']) + ', ' + str(commonPermutations[i]['Target']) + ', ' + str(commonPermutations[i]['Fuels']))

		for col in commonPermutations[0]:				# Add the names of the current permutation's experiment type, reactor, target and fuels
			newRow.append(commonPermutations[i][col])
		
		for col in commonPermutations[0]:				# Gets the row in each dataframe with the current experiment type, reactor, target and fuels
			
			try:	# Throws expections because it looks for a permutation that's not in the dataset because it has too few rows (discarded in getFinalDf)
				tempDf1 = tempDf1.groupby(col).get_group(commonPermutations[i][col])
				tempDf2 = tempDf2.groupby(col).get_group(commonPermutations[i][col])
			except:
				pass
		
		score1 = tempDf1['avg'].values[0]
		score2 = tempDf2['avg'].values[0]

		#print('score1: ' + str(score1))
		#print('score2: ' + str(score2))
		
		newRow.append(str(round(((score1 - score2) * 100), 2)) + '%')
		resultsDf.loc[len(resultsDf.index)] = newRow
	
	return resultsDf


def calculateCorrelationRatio(cat_variable, num_variable):
	fcat, _ = pandas.factorize(cat_variable)
	cat_num = np.max(fcat)+1
	y_avg_array = np.zeros(cat_num)
	n_array = np.zeros(cat_num)
	
	for i in range(0,cat_num):
		cat_measures = num_variable[np.argwhere(fcat == i).flatten()]
		n_array[i] = len(cat_measures)
		y_avg_array[i] = np.average(cat_measures)

	y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)

	numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
	denominator = np.sum(np.power(np.subtract(num_variable,y_total_avg),2))
	
	if numerator == 0:
		eta = 0.0
	else:
		eta = np.sqrt(numerator/denominator)
	
	return eta


def isCategorical(X):	# Checks if a variable is categorical even if it has numerical values (e.g. a variable with strings converted to numbers with updateTableFromDict). Tested, works
	
	if not is_numeric_dtype(X):		# If the variable has string values, it's definitely categorical
		return True
	
	possibleValues = X.unique().shape[0]
	totalValues = X.shape[0]
	#print('the variable has ' + str(possibleValues) + ' possible values, and ' + str(totalValues) + ' total values')

	if ((possibleValues / totalValues) < 0.05):		# If the possible values are less than 10% of all values, it's most likely a categorical column
		return True
	
	return False


def getCorrelationCoefficient(X, Y):		# Gets the correlation coeffient between the columns/variables X and Y
	
	if (X.equals(Y)):		# Even without this check the coefficients are still 1, which means the calculations are correct!
		coefficient = 1
		
	elif (not isCategorical(X) and not isCategorical(Y)):			# If both variables are numerical, return Pearson's coefficient
		coefficient = np.corrcoef(X, Y)[0,1]

	elif (isCategorical(X) and isCategorical(Y)):	# If both variables are categorical, return Cramer's V
		X2 = scipy.stats.chi2_contingency(pandas.crosstab(X, Y).values)[0]
		N = len(X)	# The total amount of observations are the same as one column's length
		min_dimension = min(X.nunique(), Y.nunique()) - 1		# nunique() finds the number of unique values in each column. Since it's only 1 column it returns one integer
		coefficient = np.sqrt((X2 / N) / min_dimension)

	else:														# If one variable is numerical and one is categorical
		if (not isCategorical(Y)):		# If Y is numerical (and thus X is categorical)
			coefficient = calculateCorrelationRatio(X, Y)
		else:							# Otherwise X has to be numerical and Y categorical
			coefficient = calculateCorrelationRatio(Y, X)

	return round(coefficient, 2)


def getCorrelationMatrix(dataset):	# Builds a matrix where every cell is the correlation coefficient of two columns in the dataset

	columnNames = dataset.columns.values.tolist()
	corrMatrix = pandas.DataFrame(columns=[' '] + columnNames)	# The matrix is a dataframe

	for i in range(dataset.shape[1]):		# For each column

		newRow = []
		newRow.append(columnNames[i])	# The first element of each row is the name of the column

		for j in range(dataset.shape[1]):
			coeff = getCorrelationCoefficient(dataset[columnNames[i]], dataset[columnNames[j]])
			#print('Coefficient of (' + columnNames[i] + ', ' + columnNames[j] + '): ' + str(coeff))
			newRow.append(coeff)

		corrMatrix.loc[len(corrMatrix.index)] = newRow		# Appends the new row to the correlation matrix

	return corrMatrix


def main():

	#firstModelPath = Path(sys.argv[1])		# Get model paths from command line, disabled for now
	#secondModelPath = Path(sys.argv[2])

	pandas.options.mode.chained_assignment = None	# Suppresses the annoying SettingWithCopyWarning
	kmeans_clusters = 5		# Was 15, changed because subclusters are much smaller
	topClustersNum = 2

	clusterObj = sklearn.cluster.KMeans(n_clusters=kmeans_clusters)

	dataset = Dataset(Path('data', '1800.xlsx'))
	dataset2 = Dataset(Path('data', '2100_2110.xlsx'))

	getCorrelationMatrix(dataset.df).to_excel('../correlation_matrix.xlsx')

	commonPermutations = [x for x in dataset.permutations if x in dataset2.permutations]

	'''
	print("Permutations in the first model: " + str(len(dataset.permutations)))
	print("Permutations in the second model:  " + str(len(dataset2.permutations)))
	print("Permutations in both models: " + str(len(commonPermutations)))
	for i in range(len(commonPermutations)):
		print(commonPermutations[i])
	'''

	finalDf = getFinalDf(dataset, commonPermutations, clusterObj)
	finalDf2 = getFinalDf(dataset2, commonPermutations, clusterObj)
	barChart(finalDf)

	# Build a third dataframe with the common permutations that shows the % difference between the two model for each permutation
	print(finalDf)
	print('finalDf up, now finalDf2:')
	print(finalDf2)

	resultsDf = getResultsDf(finalDf, finalDf2, commonPermutations)
	resultsDf.to_excel('../results.xlsx')
	

if __name__ == '__main__':
	main()
