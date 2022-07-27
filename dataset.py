import pandas

class Dataset:

	def __init__(self, path):
		self.df = pandas.read_excel(path, engine='openpyxl').drop(columns=['Exp SciExpeM ID', 'Experiment DOI', 'Chem Model', 'Chem Model ID'])
		self.df = self.df.drop(self.df.columns[0], axis=1)		# Removes the first unnamed column
		
		self.calculateAvg('Phi')
		self.calculateAvg('Pressure (Bar)')		# Also update the main dataset with the average values for phi, P and T
		self.calculateAvg('Temperature (K)')

		self.columns = {}
	
		self.columns['Experiment Type'] = self.df['Experiment Type'].tolist()
		self.columns['Reactor'] = self.df['Reactor'].tolist()
		self.columns['Target'] = self.df['Target'].tolist()
		self.columns['Fuels'] = self.df['Fuels'].tolist()

		self.permutations = self.getPermutations()
	
	def calculateAvg(self, column):

		for i in range(self.df.shape[0]):
			string = self.df[column].iloc[i][1:-1].replace(',', '')		# [1:-1] removes the first and last char, so '(1.0, 1.0)' becomes '1.0, 1.0'. replace() removes the comma
			minmax = [float(num) for num in string.split()]  			# List with the min and max value of the interval
			self.df[column].iloc[i] = (minmax[0] + minmax[1]) / 2		# Updates the cell with the average
		
		self.df[column] = pandas.to_numeric(self.df[column])

	def getPermutations(self):			# Returns the list of permutations (dict list) given a list of columns. Generic approach to getPermutations
		permutationsList = []
		columnNames = list(self.columns)
		rows = len(self.columns[columnNames[0]])

		for i in range(rows):		# For each row
			
			tempDict = {}

			for col in self.columns:		# For each column (col = key of columns dict = column name). columns = {ColName : [colValues], ...}
			
				tempDict[col] = self.columns[col][i]	# Add cell values to dict

			if tempDict not in permutationsList:
				permutationsList.append(tempDict)

		return permutationsList		# List of all possible permutations in the dataset (by categoric columns)

