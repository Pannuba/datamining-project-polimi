import openpyxl, pandas, sklearn, seaborn, math
from sklearn.cluster import kmeans_plusplus, KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

def standardDeviation(list):
	
	sum = 0

	for i in range(len(list)):
		sum += list[i]

	avg = sum / len(list)
	sum2 = 0

	for i in range((len(list))):
		sum2 += (list[i] - avg) * (list[i] - avg)

	stdDev = math.sqrt(sum2/len(list))
	return stdDev


def calculateAvg(dataset, column):

	for i in range(dataset.shape[0]):
		string = dataset[column].iloc[i][1:-1].replace(',', '')	 # [1:-1] toglie il primo e l'ultimo carattere, così "(1.0, 1.0)" diventa "1.0, 1.0". replace toglie la virgola
		minmax = [float(num) for num in string.split()]  # Lista col valore minimo e massimo dell'intervallo
		dataset[column].iloc[i] = (minmax[0] + minmax[1]) / 2	 # Aggiorna il valore con la media


def getStdDevOfColInCluster(dataset, clusterNumber, column):		# Per le colonne "convertite" tipo Fuel, devo passare kmeans_dataset (e quindi aggiungere la colonna Cluster anche a quello

	newList = []	   # Fare funzione?

	for i in range(dataset.shape[0]):
		if dataset["Cluster"].iloc[i] == clusterNumber:
			newList.append(dataset[column].iloc[i])

	return standardDeviation(newList)


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

dataset = original_dataset.drop(original_dataset.columns[[0,1,3,4]], axis=1)		# Toglie le colonne inutili

print("Ci sono " + str(dataset.shape[0]) + " righe nel dataset")


# Dovrei fare così per ogni colonna "strana".. hard code o modo per creare tutti i  dict/codifiche automaticamente dal file?

kmeans_dataset = dataset.drop(dataset.columns[[0,2,8,9,14]], axis=1)		# Toglie le colonne su cui non fare kmeans


fuelsDict = createDictAndUpdateTable(kmeans_dataset, "Fuels")  # Così mi tengo il dizionario per analizzare i risultati dei cluster (riconvertire da numero a string)
targetDict = createDictAndUpdateTable(kmeans_dataset, "Target")
expTypeDict = createDictAndUpdateTable(kmeans_dataset, "Experiment Type")

calculateAvg(kmeans_dataset, "Phi")
calculateAvg(kmeans_dataset, "Pressure (Bar)")
calculateAvg(kmeans_dataset, "Temperature (K)")

print(kmeans_dataset)
numClusters = 20

cluster = sklearn.cluster.KMeans(n_clusters=numClusters)
dataset["Cluster"] = cluster.fit_predict(kmeans_dataset)
print(dataset)

reduced_data = sklearn.decomposition.PCA(n_components=2).fit_transform(kmeans_dataset)
results = pandas.DataFrame(reduced_data,columns=['pca1','pca2'])
seaborn.scatterplot(x="pca1", y="pca2", hue=dataset['Cluster'], data=results)
plt.show()

# Devo prendere d0L1, L2... dei cluster con più esperimenti e fare la loro deviazione standard. Come trovo il cluster più numeroso?

clusterCount = {}	  # key:value --> cluster:numRighe, cioè quante righe ha ogni cluster

for i in range(numClusters):	 # Da 0 a 19, numero di cluster

	count = 0

	for j in range(dataset.shape[0]):

		if dataset["Cluster"].iloc[j] == i:
			count += 1
	
	clusterCount[i] = count

clusterCount = sorted(clusterCount.items(), key=lambda x: x[1], reverse=True)   # Ordina il dizionario in base al value, in ordine decrescente. Così la prima key (# cluster) è quello con più righe
print(clusterCount)
biggestCluster = clusterCount[0][0]
print(str(biggestCluster) + " è il cluster con più elementi")

for i in range(10):
	print("devStd(d0L2) nel cluster " + str(clusterCount[i][0]) + " (" + str(clusterCount[i][1]) + " righe) = " + str(getStdDevOfColInCluster(dataset, clusterCount[i][0], 'd0L2')))
	print("devStd(d1L2) nel cluster " + str(clusterCount[i][0]) + " (" + str(clusterCount[i][1]) + " righe) = " + str(getStdDevOfColInCluster(dataset, clusterCount[i][0], 'd1L2')))
	print("devStd(d0Pe) nel cluster " + str(clusterCount[i][0]) + " (" + str(clusterCount[i][1]) + " righe) = " + str(getStdDevOfColInCluster(dataset, clusterCount[i][0], 'd0Pe')))		# Ciclo for, stampa la deviazione standard dei top x cluster?
	print("devStd(d1Pe) nel cluster " + str(clusterCount[i][0]) + " (" + str(clusterCount[i][1]) + " righe) = " + str(getStdDevOfColInCluster(dataset, clusterCount[i][0], 'd1Pe')) + '\n')