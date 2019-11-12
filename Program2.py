import numpy as np
import csv

def candidateElimination():

	data = []

	csvFile = open('Data2.csv', 'r')
	reader = csv.reader(csvFile, delimiter = ',')
	
	for row in reader:
		
		data.append(np.array(row))
	


	# Convert To Numpy Array

	data = np.asarray(data, dtype = 'object')	

	X = data[:, :-1]
	Y = data[:, -1].reshape(X.shape[0], 1)	
	
	print ("\nTraining Data :")
	print (X)
	print ("\nLabels :")
	print (Y)
	
	print("\nShape Of X :")
	print (X.shape)
	print ("\nShape Of Y :")
	print (Y.shape)
	

	specificH = [" % " for _ in range(X.shape[1])]
	specificH = np.asarray(specificH, dtype = 'object')
	
	generalH = [[" ? " for _ in range(X.shape[1])] for _ in range(X.shape[1])]
	generalH = np.asarray(generalH, dtype = 'object')

	print ("\nInitial Hypothesis :")	
	print (specificH)

	print ("\nInitial General Hypothesis :")	
	print (generalH)

	# Set First Positive Example To Hypothesis	
	if Y[0] == "P":
		specificH = X[0]
	
	else:
		for i in range(Y.shape[0]):
			if Y[i] == "P":
				specificH = X[i]
				break
			
	
	
	print ("\nCandidate Elimination : ")
	
	# For Each Training Example
	for i in range(X.shape[0]):
		
		# Positive Example
		if Y[i] == "P":
			for j in range(X.shape[1]):
				if X[i][j] != specificH[j]:
					specificH[j] = '?'
				
				if specificH[j] != generalH[j][j] and generalH[j][j] != "?":
					generalH[j][j] = "?"
			
			print ("\n---------Step " + str(i + 1) + "---------\n")	
			print ("\nSpecific Set : ")
			print (specificH)
			print ("\nGeneral Set : ")
			print (generalH)
			print ("\n------------------------\n")	
		

		# Negative Example
		else:
			for j in range(X.shape[1]):
				if X[i][j] != specificH[j]:
					generalH[j][j] = specificH[j]
			
			print ("\n---------Step " + str(i + 1) + "---------\n")
			print ("\nSpecific Set : ")
			print (specificH)
			print ("\nGeneral Set : ")
			print (generalH)
			print ("\n------------------------\n")

	

	print ("\nFinal Specific Hypothesis : ")
	print (specificH)
	print ("\nFinal General Hypothesis : ")
	print (generalH)
	print ("\n")
	

candidateElimination()
