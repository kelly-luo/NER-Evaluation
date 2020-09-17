# COMP700 TEXT AND VISION INTELLIGENCE
# NER EVALUATION - ASSIGNMENT 1
# Kelly Luo (17985065)

from numpy import *
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import nltk
import os

O_TP = 0
O_FP = 0
OP_FP = 0
OL_FP = 0


P_TP = 0
P_FP = 0
PO_FP = 0
PL_FP = 0

L_TP = 0
L_FP = 0
LO_FP = 0
LP_FP = 0

TP = 0
FN = 0
FP = 0
dat_array = []
results_array = []


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


# Passing to analyse the article text and chunk with labels
def label_data(txt_writing):
    for sent in nltk.sent_tokenize(txt_writing): #for all the words in the text article
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))): #for all the connected chunks
            if hasattr(chunk, 'label'):
                if chunk.label() == 'GPE':
                    resultLabel = ' '.join(c[0] for c in chunk), "LOCATION"
                    results_array.append(resultLabel)
                else:
                    resultLabel = ' '.join(c[0] for c in chunk), chunk.label()
                    results_array.append(resultLabel)


def ner_evaluation_and_comparison():
    global TP, FP, FN, O_FN, P_FN, L_FN
    datLineCount = 0
    for entry in results_array:

        try:
            dat_array[datLineCount]
        except IndexError: # There are no more lines in the dat file
            FP += 1
            continue

        if entry[0].lower() == dat_array[datLineCount][0].lower():  # Case 1: Text is the same
            if entry[1] == dat_array[datLineCount][1] or len(entry[1]) == len(dat_array[datLineCount][1]):  # Case 1.1: Labels are the same
                correct_label_matrix_count(entry[1])
                TP += 1
                datLineCount += 1
            else:  # Case 1.1: Labels are different
                wrong_label_matrix_count(((dat_array[datLineCount][1])[:1]).capitalize(), entry[1][:1])
                FP += 1
                datLineCount += 1
        elif entry[0].lower() in dat_array[datLineCount][0].lower() and len(entry[0]) < len(dat_array[datLineCount][0]):  # Case 1: Result text is missing a word
            FP += 1
            datLineCount += 1
        elif dat_array[datLineCount][0].lower() in entry[0].lower() and len(entry[0]) > len(dat_array[datLineCount][0]):  # Case 2: Result text has an extra word
            FP += 1
            datLineCount += 1
        else:
            for x in range(0,3):
                if (datLineCount + x + 1) > len(dat_array):  # stop the loop for checking other lines in dat file if there is no more lines
                    FP += 1
                    break
                if entry[0].lower() == dat_array[datLineCount + x][0].lower():  # Case 1: Text is the same
                    if entry[1] == dat_array[datLineCount + x][1] or len(entry[1]) == len(dat_array[datLineCount + x][1]):  # Case 1.1: Labels are the same
                        correct_label_matrix_count(entry[1])
                        TP += 1
                        datLineCount = datLineCount + x + 1
                        if x > 0:  # if the results count is larger than the dat count that means it missed one
                            FN += x
                        break
                    else:  # Case 1.1: Labels are different
                        wrong_label_matrix_count(((dat_array[datLineCount][1])[:1]).capitalize(), entry[1][:1])
                        FP += 1
                        datLineCount = datLineCount + x + 1
                        if x > 0:  # if the results count is larger than the dat count that means it missed one
                            FN += x
                        break
                if x == 4: # After searching next following 5 lines, cannot find right match
                    FP += 1


def correct_label_matrix_count(results_label):
    global O_TP, P_TP, L_TP

    if results_label == 'ORGANIZATION':
        O_TP += 1
    elif results_label == 'PERSON':
        P_TP += 1
    elif results_label == 'LOCATION':
        L_TP += 1


def wrong_label_matrix_count(actual_label, pred_label):
    global OP_FP, OL_FP, PO_FP, PL_FP, LO_FP, LP_FP

    if actual_label == 'O':
        if pred_label == 'P':
            OP_FP += 1
        elif pred_label == 'L' or pred_label == 'G':
            OL_FP += 1
    elif actual_label == 'P':
        if pred_label == 'O':
            PO_FP += 1
        elif pred_label == 'L' or pred_label == 'G':
            PL_FP += 1
    elif actual_label == 'L' or actual_label == 'G':
        if pred_label == 'O':
            LO_FP += 1
        elif pred_label == 'P':
            LP_FP += 1


def calculate_recall(tp, fn):
    return "{:.2f}".format((tp / (tp + fn))*100)


def calculate_precision(tp, fp):
    return "{:.2f}".format((tp / (tp + fp))*100)


def calculate_Fvalue(precision, recall):
    return "{:.2f}".format((precision * recall) / (precision + recall))


def print_confusion_matrix():
    print("\r\n------------- Confusion Matrix -------------")
    confusionMatrix = array([[str('   '), str('Pos'), str('Neg')],
                             ['Pos','TP='+ str(TP), 'FP='+ str(FP)],
                             ['Pos','FN='+ str(FN), str('Pos')]])
    print(confusionMatrix)

    print("\r\n------------- Category Confusion Matrix -------------")
    confusionMatrix = array([[str('   '), 'O', 'P', 'L'],
                             ['O', str(O_TP), str(PO_FP) ,str(LO_FP)],
                             ['P', str(OP_FP), str(P_TP) ,str(LP_FP)],
                             ['L', str(OL_FP), str(PL_FP) ,str(L_TP)]])
    print(confusionMatrix)

    O_FP = OP_FP + OL_FP
    P_FP = PO_FP + PL_FP
    L_FP = LO_FP + LP_FP
    print("\r\nCategory Totals:")
    print("ORGANISATION ------> TP:" + str(O_TP) + " ---- FP:" + str(O_FP))
    print("PERSON ------> TP:" + str(P_TP) + " ---- FP:" + str(P_FP))
    print("LOCATION ------> TP:" + str(L_TP) + " ---- FP:" + str(L_FP))


def print_FPR():
    print("\r\n------------- FPR Calculations -------------")
    precision = calculate_precision(TP, FP)
    print("Overall Precision (P): " + str(precision))
    recall = calculate_recall(TP, FN)
    print("Overall Recall (R): " + str(recall) + "%")
    print("Overall F Value (F): " + str(calculate_Fvalue(precision, recall)) + "%")

    # print("\r\nCategory Totals:")
    # org_prec = calculate_precision(O_TP, O_FP)
    # org_recall = calculate_recall(O_TP, O_FN)
    # print("ORGANISATION ------> P:" + str(org_prec) + "% ---- R:" + (str(O_FP)) + "% ---- F:" + (str(OP_FP + OL_FP)) + "%")
    # print("PERSON ------> TP:" + str(P_TP) + " ---- FP:" + (str(PO_FP + PL_FP)))
    # print("LOCATION ------> TP:" + str(L_TP) + " ---- FP:" + (str(LO_FP + LP_FP)))

def read_dat_file(dat_writing):
    count = 0
    for line in dat_writing:

        if line.strip() == '': # if the line is empty
            count += 1
            print("---SKIPPED LINE " + str(count))
            continue

        l = line.split('(')

        # Remove the empty spaces in start and end, '\n' and '(' character
        l[0] = l[0].strip()

        try:
            l[1] = l[1].strip()
            l[1] = (l[1])[:-1]
            if 'GPE' in l[1]:  # Change GPE to LOCATION when storing into dat_array
                l[1] = 'LOCATION'
        except IndexError: # Missing label
            continue

        dat_array.append(l)
        count += 1
        print("LINE " + str(count) + " EXTRACTED")


# Set to your own dataset path with all the dat and txt files in the same folder
datasetPath = 'C:/Users/LuoKe/OneDrive/Documents/EntireDataset/'
files = array(os.listdir(datasetPath))

fileCount = 0;
while True:
    if fileCount >= 554:
        break
    currentFile = files[fileCount].split('.')
    if fileCount + 1 <= 560:
        nextFile = files[fileCount + 1].split('.')
    if files[fileCount].endswith('.dat'):
        if currentFile[0] == nextFile[0] and nextFile[1] == 'txt': # check if the next matching text file matches
            print("\r\n\r\n------ Processing file: " + currentFile[0] + " ------ \r\n")

            try:
                datFile = open(os.path.join(datasetPath, files[fileCount]), 'r', encoding='UTF8')
                datWriting = datFile.readlines()
            except UnicodeDecodeError:
                datFile = open(os.path.join(datasetPath, files[fileCount]), 'r')
                datWriting = datFile.readlines()

            try:
                txtFile = open(os.path.join(datasetPath, files[fileCount + 1]), 'r', encoding='UTF8')
                txtWriting = txtFile.read()
            except UnicodeDecodeError:
                txtFile = open(os.path.join(datasetPath, files[fileCount + 1]), 'r')
                txtWriting = txtFile.read()

            read_dat_file(datWriting)
            label_data(txtWriting)
            ner_evaluation_and_comparison()

            dat_array.clear()
            results_array.clear()

            fileCount += 2
        else:
            print("SKIPPED " + currentFile[0])
            fileCount += 2

print_confusion_matrix()
print_FPR()