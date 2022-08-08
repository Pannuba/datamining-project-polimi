import plotly.graph_objs as go, numpy as np

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


def barChart(finalDf):		# Shows the bar chart for only one model
	
	lissst = []

	for i in range(finalDf.shape[0]):
		lissst.append(i)

	#fig = go.Figure(data=go.Heatmap(x=finalDf['Fuels'], y=finalDf['Target'], z=finalDf['avg'], text=[finalDf['Experiment Type'], finalDf['Reactor'], finalDf['Target'], finalDf['Fuels']], texttemplate='%{text}'))
	avgChart = go.Bar(	name='Average', marker_color='lightskyblue', x=lissst, y=finalDf['avg'].round(4), error_y=dict(type='data', array=finalDf['std'].round(4)),
						customdata=np.stack((finalDf['Experiment Type'], finalDf['Reactor'], finalDf['Target'], finalDf['Fuels']), axis=-1),
						hovertemplate='Exp. Type: %{customdata[0]}<br>Reactor: %{customdata[1]}<br>Target: %{customdata[2]}<br>Fuels: %{customdata[3]}<br>Average: %{y}')
	medChart = go.Bar(	name='Median', marker_color='salmon', x=lissst, y=finalDf['median'].round(4),
						customdata=np.stack((finalDf['Experiment Type'], finalDf['Reactor'], finalDf['Target'], finalDf['Fuels']), axis=-1),
						hovertemplate='Exp. Type: %{customdata[0]}<br>Reactor: %{customdata[1]}<br>Target: %{customdata[2]}<br>Fuels: %{customdata[3]}<br>Median: %{y}')
	fig = go.Figure(data=[avgChart, medChart])
	fig.update_layout(title_text='Average, median and standard deviation of the score for each permutation in the model', barmode='group')
	
	fig.show()


def barChartBoth(finalDf, finalDf2, modelName1, modelName2):	# Shows the bar chart comparing two datasets
	
	lissst = []

	for i in range(finalDf.shape[0]):
		lissst.append(i)

	avgChart = go.Bar(	name=modelName1+' avg', marker_color='lightskyblue', x=lissst, y=finalDf['avg'].round(4), error_y=dict(type='data', array=finalDf['std'].round(4)),
						customdata=np.stack((finalDf['Experiment Type'], finalDf['Reactor'], finalDf['Target'], finalDf['Fuels']), axis=-1),
						hovertemplate='Exp. Type: %{customdata[0]}<br>Reactor: %{customdata[1]}<br>Target: %{customdata[2]}<br>Fuels: %{customdata[3]}<br>Average: %{y}')
	
	avgChart2 = go.Bar(	name=modelName2+' avg', marker_color='salmon', x=lissst, y=finalDf2['avg'].round(4), error_y=dict(type='data', array=finalDf2['std'].round(4)),
						customdata=np.stack((finalDf2['Experiment Type'], finalDf2['Reactor'], finalDf2['Target'], finalDf2['Fuels']), axis=-1),
						hovertemplate='Exp. Type: %{customdata[0]}<br>Reactor: %{customdata[1]}<br>Target: %{customdata[2]}<br>Fuels: %{customdata[3]}<br>Average: %{y}')

	fig = go.Figure(data=[avgChart, avgChart2])
	fig.update_layout(title_text='Average and standard deviation of the score for each permutation in the models', barmode='group')
	
	fig.show()
