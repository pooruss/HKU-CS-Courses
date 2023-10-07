# -*- coding: UTF-8 -*-
from math import log
import operator
import csv

from Readdata import Readdata
import numpy as np
from Evaluation import Evaluation


class ID3:

    def __init__(self, dataset):
        dataSet_All = dataset
        self.labels = dataSet_All[0]
        self.dataSet = dataSet_All[1:]



    """
    Function description: calculate the empirical entropy of a given dataset (Shannon entropy)
    Parameters:
        dataSet 
    Returns:
        shannonEnt
    """


    def calcShannonEnt(self, dataSet):
        num_entires = len(dataSet)  # Returns the number of rows in the dataset
        label_counts = {}  # Save the dictionary of the number of occurrences of each label
        for featVec in dataSet:  # Make statistics on each group of feature vectors and traverse each row in the list
            current_label = featVec[-1]  # Extract Label information
            if current_label not in label_counts.keys():  # If the label is not placed in the dictionary for counting times, add it
                label_counts[current_label] = 0
            label_counts[current_label] += 1  # Label counting
        shannonEnt = 0.0  # Empirical entropy)
        for key in label_counts:  # Calculate Shannon entropy
            prob = float(label_counts[key]) / num_entires  # Probability of selecting the label
            shannonEnt -= prob * log(prob, 2)  # Calculate by formula
        return shannonEnt  # Return to empirical entropy (Shannon entropy)


    """
    Function description: divide the data set according to the given characteristics
    Parameters:
        DataSet - The dataset to be divided
    
        Axis - characteristics of partitioned data sets
    
        Value - the value of the characteristic to be returned
    """


    def splitDataSet(self, dataSet, axis, value):
        retDataSet = []  # Create a list of returned data sets
        for featVec in dataSet:  # Traversal data set
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]  # Remove axis feature
                reducedFeatVec.extend(featVec[axis + 1:])  # Add qualified data to the returned dataset
                retDataSet.append(reducedFeatVec)
        return retDataSet  # Return the partitioned dataset


    """
    Function description: select the best feature
    
    Parameters:
    
    DataSet - Dataset
    
    Returns:
    
    BestFeature - Index value of the (optimal) feature with the largest information gain
    """


    def chooseBestFeatureToSplit(self, dataSet):
        numFeatures = len(dataSet[0]) - 1  # Characteristic quantity
        baseEntropy = self.calcShannonEnt(dataSet)  # Calculate Shannon entropy of data set
        bestInfoGain = 0.0  # Information gain
        bestFeature = -1  # Index value of optimal feature
        for i in range(numFeatures):  # Traverse all features
            # Get all the ith characteristics of the dataset
            featList = [example[i] for example in dataSet]
            uniqueVals = set(featList)  # Create set set {}, element cannot be repeated
            newEntropy = 0.0  # Empirical conditional entropy
            for value in uniqueVals:  # Calculate information gain
                subDataSet = self.splitDataSet(dataSet, i, value)  # SubDataSet divided subset
                prob = len(subDataSet) / float(len(dataSet))  # Calculate the probability of subset
                newEntropy += prob * self.calcShannonEnt(
                    subDataSet)  # Calculate empirical conditional entropy according to the formula
            infoGain = baseEntropy - newEntropy  # information gain
            print("第%d个特征的增益为%.3f" % (i, infoGain))  # Print information gain for each feature
            if (infoGain > bestInfoGain):  # Calculate information gain
                bestInfoGain = infoGain  # Update the information gain and find the maximum information gain
                bestFeature = i  # Record the index value of the feature with the largest information gain
        return bestFeature  # Returns the index value of the feature with the largest information gain


    """
    Function description: count the most elements (class labels) in the classList
    
    Parameters:
    
    ClassList - class label list
    
    Returns:
    
    SortedClassCount [0] [0] - Most elements (class label)
    """


    def majorityCnt(self, classList):
        classCount = {}
        for vote in classList:  # Count the number of occurrences of each element in the classList
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),
                                  reverse=True)  # Sort by dictionary value in descending order
        return sortedClassCount[0][0]  # Returns the element with the most occurrences in the classList


    """
    Function description: recursively build a decision tree
    
    Parameters:
    
    DataSet - training dataset
    
    Labels - Category attribute label
    
    FeatLabels - stores the best feature labels selected
    
    Returns:
    
    MyTree - Decision Tree
    """


    def createTree(self, dataSet, labels, featLabels):
        classList = [example[-1] for example in dataSet]  # Get classification label
        if classList.count(classList[0]) == len(classList):  # If the categories are identical, stop dividing
            return classList[0]
        if len(dataSet[
                   0]) == 1:  # Return the class label with the highest number of occurrences after traversing all features
            return self.majorityCnt(classList)
        bestFeat = self.chooseBestFeatureToSplit(dataSet)  # Select the best feature
        bestFeatLabel = labels[bestFeat]  # Label of optimal feature
        featLabels.append(bestFeatLabel)
        myTree = {bestFeatLabel: {}}  # Label generation tree based on optimal features
        del (labels[bestFeat])  # Delete used feature labels
        featValues = [example[bestFeat] for example in
                      dataSet]  # Get the attribute values of all the best features in the training set
        uniqueVals = set(featValues)  # Remove duplicate attribute values
        for value in uniqueVals:
            subLabels = labels[:]
            # The function createTree () is called recursively to traverse the features and create a decision tree.
            myTree[bestFeatLabel][value] = self.createTree(self.splitDataSet(dataSet, bestFeat, value), subLabels, featLabels)
        return myTree


    """
    Function description: use decision tree to perform classification
    
    Parameters:
    
    InputTree - The generated decision tree
    
    FeatLabels - stores the best feature labels selected
    
    TestVec - test data list, with the order corresponding to the optimal feature label
    
    Returns:
    
    ClassLabel - Classification result
    """


    def classify(self, inputTree, featLabels, testVec):
        firstStr = next(iter(inputTree))  # Get decision tree nodes
        secondDict = inputTree[firstStr]  # Next dictionary
        featIndex = featLabels.index(firstStr)
        for key in secondDict.keys():
            if testVec[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = self.classify(secondDict[key], featLabels, testVec)
                else:
                    classLabel = secondDict[key]
        return classLabel


    def id3(self):
        # dataSet = createDataSet()
        featLabels = []


        myTree = self.createTree(self.dataSet, self.labels, featLabels)
        print(featLabels[1])

        print(myTree)
        testVec = ['1', '0']  # testing
        result = self.classify(myTree, featLabels, testVec)
        if result == '1':
            print('lending')
        if result == '0':
            print('no lending')

        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 1])

        evaluation=Evaluation()
        F1=evaluation.F1(y_true,y_pred)
        print(F1)

