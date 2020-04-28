'''
Author: James Lee Hu
University Information: University of Arizona, Management of Information Systems major, Class of 2022
Date: April 28, 2020

Code Objective: This code demonstrates an LSTM's ability to predict the price of one specific stock without
    relying on news data and sentimental analysis through several other data sources. It is the subject of the final
    research project for Dr. Hsinchun Chen's MIS 464 Data Analytics class in the spring semester of 2020.

Special thanks to JingSong Wu, my project partner. Thanks to his hardwork and resourcefulness, this model was provided
the comprehensive dataset on several macroeconomic metrics it needed to make its predictions. He also helped a lot with
the project presentation and the paper. :)

If Dr. Chen is reading this, please give us a good grade. :D
'''

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import csv

import matplotlib.pyplot as plt


class LSTM(nn.Module):
    def __init__(self, input_size=7, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def splitData(data, percentSplit=0.5):
    '''
    Splits data into training and testing datasets
    :param data: Aggregate dataset to be splitted
    :param percentSplit: Percent split between training and testing datasets.
        Training dataset will be the percent split while testing is 1 - percentsplit
    :return: Training data and testing data.
    '''
    return data[: int(len(data) * percentSplit)], data[int(len(data) * percentSplit):]


def normalizeData(data, inverse=False):
    '''
    Normalizes a dataset
    :param data: Dataset to be normalized
    :param inverse: If true, it will denormalize the dataset using the current normalizer
    :return: Normalized or de-normalized data
    '''

    global normalizer

    if inverse:
        return normalizer.inverse_transform(np.array(data).reshape(-1, 1))
    else:
        output = normalizer.fit_transform(data.reshape(-1, 1))
        normalizer = normalizer.fit(data.reshape(-1, 1))
        return output


def setPltWindow(windName="Graph", windX=0, windY=0, xLabel="X Variable", yLabel="Y Variable"):
    '''
    Initialized Matplotlib window for plotting
    :param windName: Window name
    :param windX: Window width
    :param windY: Window height
    :param xLabel: X axis label
    :param yLabel: Y axis label
    :return: None
    '''
    figSize = plt.rcParams["figure.figsize"]
    figSize[0] = windX
    figSize[1] = windY
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.autoscale(axis='y', tight=True)
    plt.rcParams["figure.figsize"] = figSize

    plt.title(windName)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    plt.autoscale(axis='x', tight=True)


def create_inout_sequences(input_data, tw):
    '''
    Splits input data into an input/output format.
        The format consists of a list where each element is a input (data with the size of the LSTM's input size) and \
        a label (data with the size of the LSTM's output size)
    :param input_data: Data to be formatted
    :param tw: Training window
    :return: Training input/output Sequence
    '''
    inout_seq = []
    L = len(input_data)
    # Iterates through data, stopping at a training window length away from the data's end
    for i in range(L - tw):
        # Training sequence is the input data
        train_seq = input_data[i:i + tw].astype(np.float32)

        # Training label set to the next S&P Close value (position in dataset hard-coded in)
        train_label = input_data[i + tw:i + tw + 1][0][-1].astype(np.float32)

        if torch.cuda.is_available():
            inout_seq.append((torch.tensor(train_seq, device=torch.device("cuda")),
                              torch.tensor(train_label, device=torch.device("cuda"))))
        else:
            inout_seq.append((torch.tensor(train_seq), torch.tensor(train_label)))
    return inout_seq


def trainModelFromScratch(trainIOSeq, epochLimit):
    '''
    Trains a LSTM model from scratch
    :param trainIOSeq: Training input/output sequence
    :param epochLimit: Stops training once epoch limit is reached
    :return: trained model and a list of lost values
    '''
    model = LSTM()
    if torch.cuda.is_available():
        model.cuda()
    lossFunction = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    loss = []

    for epoch in range(epochLimit):
        for seq, labels in trainIOSeq:
            optimizer.zero_grad()

            if torch.cuda.is_available():
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size, device=torch.device("cuda")),
                                     torch.zeros(1, 1, model.hidden_layer_size, device=torch.device("cuda")))
            else:
                model.hidden_cell = (
                torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))

            yPredicted = model(seq)

            singleLoss = lossFunction(yPredicted, labels)
            singleLoss.backward()
            optimizer.step()

        loss.append(singleLoss.item())
        print(f'epoch: {epoch:3} loss: {singleLoss.item():10.8f}')

    print(f'epoch: {epoch:3} loss: {singleLoss.item():10.10f}')
    return model, loss


def testModel(model, data, totalPredictions, trainingWindow, allDataTest= False):
    '''
    Tests model on
    :param model: Trained LSTM model to be tested
    :param data: Testing data to be tested on
    :param totalPredictions: Total number of predictions planned to be made
    :param trainingWindow: Training window
    :param allDataTest: If true, method will iterate through all data given instead of being limited
        to the totalPredictions number
    :return: De-normalized predictions and de-normalized actual data
    '''
    model.eval()

    if allDataTest:
        testData = data.tolist()
        totalPredictions = len(data) - trainingWindow
    else:
        testData = data[-totalPredictions - trainingWindow:].tolist()

    predictions = []
    actual = []

    for i in range(totalPredictions):
        seq = torch.FloatTensor(testData[-totalPredictions - trainingWindow + i: - totalPredictions + i])

        if torch.cuda.is_available():
            seq = torch.tensor(seq, device=torch.device("cuda"))

        with torch.no_grad():
            if torch.cuda.is_available():
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size, device=torch.device("cuda")),
                                     torch.zeros(1, 1, model.hidden_layer_size, device=torch.device("cuda")))
            else:
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                     torch.zeros(1, 1, model.hidden_layer_size))

            predictions.append(model(seq).item())
            actual.append(seq[-1][-1].item())

    return normalizeData(predictions, inverse=True), normalizeData(actual, inverse=True)


def stackData(dataList=[]):
    '''
    Stacks several datasets into one dataset
    :param dataList: List of datasets
    :return: Stacked dataset

    Note: This can be done using np.stack. I just don't know how.
    '''
    output = []
    for i in range(len(dataList[0])):
        output.append([])

    for dataset in dataList:
        for elementListIndex in range(len(dataset)):
            for element in dataset[elementListIndex]:
                output[elementListIndex].append(element)

    return np.array(output)


trainingWindow = 12
trainTestSplit = 0.8
normalizer = MinMaxScaler(feature_range=(-1, 1))

learningRate = 0.0001
epochLimit = 1

datapath = './Aggregate Data Trimmed and Deleted.csv'

csvData = {}

print("Reading CSV")
with open(datapath) as csvfile:
    reader = csv.reader(csvfile)
    header = True
    Headers = []
    for row in reader:
        if header:
            header = False
            for element in row:
                csvData[element] = []
            Headers = row
        else:
            for elementIndex in range(len(row)):
                if row[elementIndex] == "":
                    csvData[Headers[elementIndex]].append(csvData[Headers[elementIndex]][-1])
                else:
                    csvData[Headers[elementIndex]].append([float(row[elementIndex])])

trainingDataList = []
aggregateDataList = []
close = csvData["S&P Close"]

print("Preprocessing Data")
for dataset in csvData:
    trainData, testData = splitData(csvData[dataset], percentSplit=trainTestSplit)
    trainingDataList.append(normalizeData(np.array(trainData)))
    aggregateDataList.append(normalizeData(np.array(csvData[dataset])))

trainingData = stackData(trainingDataList)
aggregateData = stackData(aggregateDataList)

print("Creating IO Sequence")
trainIOSeq = create_inout_sequences(trainingData, trainingWindow)

model, loss = trainModelFromScratch(trainIOSeq, epochLimit)

predictions, actual = testModel(model, aggregateData,
                                totalPredictions=aggregateData.shape[0] - trainingData.shape[0],
                                trainingWindow=trainingWindow,
                                allDataTest= True)


with open('./loss.txt', 'w') as writefile:
    for losspoint in loss:
        writefile.write(str(losspoint) + "\n")

with open('./predictions.txt', 'w') as writefile:
    for i in range(len(predictions)):
        writefile.write(str(predictions[i][0]) + " ")
        writefile.write(str(actual[i][0]) + "\n")

setPltWindow("Stock Close Prices", windX=13, windY=5, xLabel="Date", yLabel="Stock Close")

# This portion of the code needs to be changed depending on what the output format/dimension is
#  and what the user wants to display
plt.plot(close[trainingWindow:])
x = np.arange(0, aggregateData.shape[0] - trainingWindow, 1)
plt.plot(x, predictions)
plt.show()
