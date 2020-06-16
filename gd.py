import csv
import os

import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from norm import normalisation


def loadData(fileName, inputVariabName1, inputVariabName2, outputVariabName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    selectedVariable1 = dataNames.index(inputVariabName1)
    selectedVariable2 = dataNames.index(inputVariabName2)
    inputs = [[float(data[i][selectedVariable1])] for i in range(len(data))]
    selectedOutput = dataNames.index(outputVariabName)
    outputs = [[float(data[i][selectedVariable2]), float(data[i][selectedOutput])] for i in range(len(data))]

    return inputs, outputs


def main():
    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, 'data', 'tshirts.csv')

    inputs, outputs = loadData(filePath, 'temperature', 'femaleTshirts', 'maleTshirts')
    print(inputs)
    print(outputs)
    print('in:  ', inputs[:5])
    print('out: ', outputs[:5])
    # maximum tshirts sold

    sum_fem = 0
    sum_male = 0
    for f in range(len(outputs)):
        sum_fem += outputs[f][0]
        sum_male += outputs[f][1]
    maximum = "females"
    if max(sum_fem, sum_male) == sum_male:
        maximum = "males"

    print("Cele mai multe tricouri ( " + str(max(sum_fem, sum_male)) + " ) au fost tricourile pentru  " + maximum)


    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if not i in trainSample]
    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    trainOutputsMaleTshirts = [[trainOut[1]] for trainOut in trainOutputs]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]
    testOutputsMaleTshirts = [[testOut[1]] for testOut in testOutputs]
    print(trainInputs)
    print(trainOutputsMaleTshirts)


    # data normalisation for train and test data
    # using my normalisation

    trainInputs, testInputs = normalisation(trainInputs, testInputs)
    trainOutputsMaleTshirts, testOutputsMaleTshirts = normalisation(trainOutputsMaleTshirts, testOutputsMaleTshirts)

    xx = [[el] for el in trainInputs]
    regressor = linear_model.SGDRegressor()
    # make SGDregressor into GD Batch regressor

    regressor.fit(xx, trainOutputs)
    w0, w1, w2 = regressor.intercept_[0], regressor.coef_[0], regressor.coef_[1]
    print('the learnt model with sklearn norm: f(x) = ', w0, ' + ', w1, ' * x', ' +', w2, ' * x^2')

    computedTestOutputs = regressor.predict([x for x in testInputs])

    error = mean_squared_error(testOutputs, computedTestOutputs)
    print("prediction error (tool): ", error)


main()
