# # Introduction
# This example illustrate a very simple linear regression couple with a cross validation mechanism.This example uses the only the first feature of the `diabetes` dataset, in order to illustrate a two-dimensional plot of this regression technique. The
# straight line can be seen in the plot, showing how linear regression attempts to draw a straight line that will best minimize the residual sum of squares between the observed responses in the dataset, and the responses predicted by
# the linear approximation.
#
# The coefficients, the residual sum of squares and the variance score are also calculated.
# This example is inspired from Jaques Grobler's.

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

#####################################
# main program                      #
#####################################
permutation = int(sys.argv[1])
#permutation = int(3)

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
diabetesX = diabetes.data[:, np.newaxis, 2]

chunkSize = int(round(len(diabetesX)/5))

# Split the data into training/testing sets
diabetesXTest = diabetesX[permutation*chunkSize: (permutation+1)*chunkSize]
diabetesXTrain = np.delete(diabetesX, list(range(permutation*chunkSize,(permutation+1)*chunkSize)),0)

# Split the targets into training/testing sets
diabetesYTest = diabetes.target[permutation*chunkSize: (permutation+1)*chunkSize]
diabetesYTrain = np.delete(diabetes.target[:],list(range(permutation*chunkSize,(permutation+1)*chunkSize)))


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetesXTrain, diabetesYTrain)

# The mean squared error
meanSquareError = "Mean squared error: %.2f"% np.mean((regr.predict(diabetesXTest) - diabetesYTest) ** 2)
print(meanSquareError)
# Explained variance score: 1 is perfect prediction
varianceScore = 'Variance score: %.2f' % regr.score(diabetesXTest, diabetesYTest)
print(varianceScore)

# Plot outputs
plt.scatter(diabetesXTest, diabetesYTest,  color='black')
plt.plot(diabetesXTest, regr.predict(diabetesXTest), color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

Linreg = "Diabetes_Linreg_" + str(permutation) + ".png"
plt.savefig("results/" + Linreg)

outFileName = "Diabetes_Linreg_metrics_" + str(permutation) + ".txt"
with open("results/" + outFileName, "a") as w:
    w.write(meanSquareError+"\n")
    w.write(varianceScore)
    w.close()