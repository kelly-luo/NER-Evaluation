# COMP700 TEXT AND VISION INTELLIGENCE - ASSIGNMENT 1
# Kelly Luo (17985065)

from numpy import *
import numpy as np
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import nltk

TP = 0;
FN = 0;
FP = 0;

def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []
    # print(chunked)
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    if continuous_chunk:
        named_entity = " ".join(current_chunk)
        if named_entity not in continuous_chunk:
            continuous_chunk.append(named_entity)

    return continuous_chunk


def calculateRecall():
    return (TP/(TP+FN))*100


def calculatePrecision():
    return (TP/(TP+FP))*100


def calculateFvalue(precision, recall):
    return (precision * recall) / (precision + recall)


def printConfusionMatrix():
    print("------------- Confusion Matrix -------------")
    confusionMatrix = array([[' ','Pos', 'Neg'],
                       ['Pos','TP='+ str(TP), 'FP='+ str(FP)],
                       ['Neg','FN='+ str(FN), 'N/A']])
    print(confusionMatrix)
    print("TP = " + str(TP))
    print("FN = " + str(FN))
    print("FP = " + str(FP))

def printFPR():
    print("------------- FPR Calculations -------------")
    precision = calculatePrecision()
    print("Precision is: " + str(precision))

    recall = calculateRecall()
    print("Recall is: " + str(recall))
    print("F Value is: " + str(calculateFvalue(precision, recall)))

path = 'C:\\Users\\LuoKe\\OneDrive\\Documents\\17985065_1.txt'
text  = open(path)
# print(text.read())

txtWriting = open(path, 'r', encoding='UTF8').read()

path2 = 'C:\\Users\\LuoKe\\OneDrive\\Documents\\17985065_1.dat'
dat  = open(path2)

datWriting = open(path2, 'r', encoding='UTF8').read()
print('\r\n')

# txt = "Jacinda Ardern is the Prime Minister of New Zealand but Roenzo isn't."
# print (get_continuous_chunks(txt))

dat_array = []
count = 0

lines = dat.readlines()

for line in lines:
    if line[-1] == '\n':
        line = line[:-2] # remove the ) and \n
    else:
        line = line[:-1] # Last line just remove )

    dat_array.append(tuple(part for part in (line.split(' (')) if part))


# with open(path2) as datFile:
#     while True:
#         count += 1
#
#         # Get next line from file
#         line = datFile.readline()
#
#         # if line is empty, end of file is reached
#         if not line:
#             line = line[:-1]
#             count = 0
#             break
#
#         line = line[:-2]  # remove the ) and new line
#         dat_array.append(tuple(part for part in (line.split(' (')) if part))

# Passing to analyse the article text and chunk with labels

results_array = []
for sent in nltk.sent_tokenize(txtWriting): #for all the words in the text article
    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))): #for all the connected chunks
        if hasattr(chunk, 'label'):
            if chunk.label() == 'GPE':
                # print(' '.join(c[0] for c in chunk) + " LOCATION")
                resultLabel = ' '.join(c[0] for c in chunk) , "LOCATION"
                results_array.append(resultLabel)
            else:
                # print(' '.join(c[0] for c in chunk) + chunk.label())
                resultLabel = ' '.join(c[0] for c in chunk) , chunk.label()
                results_array.append(resultLabel)

datLineCount = 0
for entry in results_array:
    if entry[0].lower() == dat_array[datLineCount][0].lower():  # Case 1: Text is the same
        if entry[1] == dat_array[datLineCount][1] or len(entry[1]) == len(dat_array[datLineCount][1]):  # Case 1.1: Labels are the same
            TP += 1
            datLineCount += 1
        else:  # Case 1.1: Labels are different
            FP += 1
            datLineCount += 1
    elif entry[0].lower() in dat_array[datLineCount][0].lower() and len(entry[0]) < len(dat_array[datLineCount][0]):  # Case 1: Result text is missing a word
        FP += 1
        datLineCount += 1
    elif dat_array[datLineCount][0].lower() in entry[0].lower() and len(entry[0]) > len(dat_array[datLineCount][0]):  # Case 2: Result text has an extra word
        FP += 1
        datLineCount += 1
    else:
        for x in range(0,5):
            if (datLineCount + x + 1) > len(dat_array): # stop the loop for checking other lines in dat file if there is no more lines
                FP += 1
                break
            if entry[0].lower() == dat_array[datLineCount + x][0].lower():  # Case 1: Text is the same
                if entry[1] == dat_array[datLineCount + x][1] or len(entry[1]) == len(dat_array[datLineCount + x][1]):  # Case 1.1: Labels are the same
                    TP += 1
                    datLineCount = datLineCount + x + 1
                    if(x > 0): # if the results count is larger than the dat count that means it missed one
                        FN += x
                    break
                else:  # Case 1.1: Labels are different
                    FP += 1
                    datLineCount = datLineCount + x + 1
                    if(x > 0): # if the results count is larger than the dat count that means it missed one
                        FN += x
                    break
            if x == 4: # After searching next following 5 lines, cannot find right match
                FP += 1


printConfusionMatrix()
printFPR()