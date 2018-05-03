from __future__ import print_function

__author__ = 'rudramurthy'


class Hypothesis():

    def __init__(self):
        self.targetOutput = []
        self.probabilityScore = 0.0
        self.currentCost = 0.0
        self.output = []
        self.predictions = []

    def setPredictions(predictions):
        for i in range(len(predictions)):
            self.predictions.append(predictions[i])

    def createPredictions(predictions):
        self.predictions.append(predictions)

    def getPredictions():
        return self.predictions

    def createTargetWord(targetWord):
        self.targetOutput.append(targetWord)

    def copyTargetWord(targetWord):
        for i in range(targetWord):
            self.targetOutput.append(targetWord[i])

    def setTargetWord(targetWord):
        self.output.append(targetWord)

    def getTargetIndex():
        return self.output[0]

    def getTarget():
        return self.output

    def insertProbability(probability):
        self.probabilityScore = probability

    def getProbability():
        return self.probabilityScore

    def insertCost(cost, previousStateCost):
        self.currentCost = previousStateCost + cost

    def getCost():
        return self.currentCost
