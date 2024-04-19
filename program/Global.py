# This module features all global variables
import tkinter as tk

# Global constants
bgDark = "#a0a0a0"
bgLight = "#d0d0d0"
bgBlue = "#a0d0ff"

fontTitle = ("Arial", 16)
fontLabel = ("Arial", 12)

# Global numeric variables
numRows = None # height of the input layer
numCols = None # width of the input layer

outputLayerSize = None   # size of the output layer

numberTestRecords = None
numberTrainingRecords = None

# Global objects
neuronNet = None
gui = None

allTrainingPattern = []
outputLayerLabels = None # Labels of output layer neurons
allTestPattern = []

test_file_name = None
training_file_name = None
