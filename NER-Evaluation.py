'''
COMP700 TEXT AND VISION INTELLIGENCE
ASSIGNMENT 1 - NAMED ENTITY RECOGNITION (NER)
Kelly Luo (17985065)
'''
from numpy import *
import nltk
import os

# Variables for True Positive and False Positive for ORGANIZATION label
O_TP = 0
O_FP = 0
OP_FP = 0
OL_FP = 0

# Variables for True Positive and False Positive for PERSON label
P_TP = 0
P_FP = 0
PO_FP = 0
PL_FP = 0

# Variables for True Positive and False Positive for LOCATION label
L_TP = 0
L_FP = 0
LO_FP = 0
LP_FP = 0

TP = 0
FN = 0
FP = 0
dat_array = []
results_array = []


# Method that analyses the article text and chunk with category labels and adds into an array for the results
def chunk_label_data(txt_writing):
    for sent in nltk.sent_tokenize(txt_writing):  # for all the words in the text article
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):  # for all the connected chunks
            if hasattr(chunk, 'label'):
                if chunk.label() == 'GPE':  # changing label to be LOCATION if it is GPE
                    resultLabel = ' '.join(c[0] for c in chunk), "LOCATION"
                    results_array.append(resultLabel)
                else:
                    resultLabel = ' '.join(c[0] for c in chunk), chunk.label()
                    results_array.append(resultLabel)


# Method that compares the NER evaluation between each match between the labelled results array (predicted)
# with the dat file (actual)
def ner_evaluation_and_comparison():
    global TP, FP, FN
    datLineCount = 0
    for entry in results_array:
        try:
            dat_array[datLineCount]
        except IndexError:  # There are no more lines in the dat file
            FP += 1
            continue

        # Case 1: Both Predicted and Actual text are the same
        if entry[0].lower() == dat_array[datLineCount][0].lower():

            # Case 1.1:  Both Predicted and Actual text are the same BUT Labels are the same
            if entry[1] == dat_array[datLineCount][1] or len(entry[1]) == len(dat_array[datLineCount][1]):
                correct_label_matrix_count(entry[1])
                TP += 1
                datLineCount += 1

            # Case 1.2: Both Predicted and Actual text are the same BUT Labels are different
            else:
                wrong_label_matrix_count(((dat_array[datLineCount][1])[:1]).capitalize(), entry[1][:1])
                FP += 1
                datLineCount += 1

        # Case 2: Predicted text is missing a word from the Actual text
        elif entry[0].lower() in dat_array[datLineCount][0].lower() and len(entry[0]) < len(dat_array[datLineCount][0]):
            FP += 1
            datLineCount += 1

        # Case 3: Predicted text has an extra word compared to the Actual text
        elif dat_array[datLineCount][0].lower() in entry[0].lower() and len(entry[0]) > len(dat_array[datLineCount][0]):
            FP += 1
            datLineCount += 1

        # Case 4: The current comparison between predicted text and actual test is not a match
        else:
            for x in range(0,3):  # Loop for the next following 3 lines to see if there is a match in text
                if (datLineCount + x + 1) > len(dat_array):  # Stop the loop if there is no more lines
                    FP += 1
                    break

                # Case 4.1: Both Predicted and Actual text are the same
                if entry[0].lower() == dat_array[datLineCount + x][0].lower():

                    # Case 4.2: Both Predicted and Actual text are the same BUT Labels are the same
                    if entry[1] == dat_array[datLineCount + x][1] or len(entry[1]) == len(dat_array[datLineCount + x][1]):
                        correct_label_matrix_count(entry[1])
                        TP += 1
                        datLineCount = datLineCount + x + 1

                        # Case 5: When lines in dat file is skipped, this mean NER was not identified
                        if x > 0:
                            FN += x
                        break

                    # Case 4.3: Both Predicted and Actual text are the same BUT Labels are different
                    else:
                        wrong_label_matrix_count(((dat_array[datLineCount][1])[:1]).capitalize(), entry[1][:1])
                        FP += 1
                        datLineCount = datLineCount + x + 1

                        # Case 5: When lines in actual values is skipped, this means that NER did not identify
                        if x > 0:
                            FN += x
                        break

                # Case 6: There are no matches in the following 3 lines in actual values therefore aditional identification
                if x == 4:
                    FP += 1


# Method to count the correct category labels
def correct_label_matrix_count(results_label):
    global O_TP, P_TP, L_TP

    if results_label == 'ORGANIZATION':
        O_TP += 1
    elif results_label == 'PERSON':
        P_TP += 1
    elif results_label == 'LOCATION':
        L_TP += 1


# Method to count the incorrect category labels
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


# Method to calculate Recall in percentage
def calculate_recall(tp, fn):
    return (tp / (tp + fn))*100


# Method to calculate Precision in percentage
def calculate_precision(tp, fp):
    return (tp / (tp + fp))*100


# Method to calculate F Value in percentage
# Note: Beta value is 1
def calculate_Fvalue(precision, recall):
    return (precision * recall)/(precision + recall)


# Method to print overall confusion matrix and category matrix
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


# Method to print the overall FPR calculations
def print_FPR():
    print("\r\n------------- FPR Calculations -------------")
    precision = calculate_precision(TP, FP)
    print("Overall Precision (P): " + str("{:.2f}".format(precision)) + "%")
    recall = calculate_recall(TP, FN)
    print("Overall Recall (R): " + str("{:.2f}".format(recall)) + "%")
    print("Overall F Value (F): " + str("{:.2f}".format(calculate_Fvalue(precision, recall))) + "%")

    # print("\r\nCategory Totals:")
    # org_prec = calculate_precision(O_TP, O_FP)
    # org_recall = calculate_recall(O_TP, O_FN)
    # print("ORGANISATION ------> P:" + str(org_prec) + "% ---- R:" + (str(O_FP)) + "% ---- F:" + (str(OP_FP + OL_FP)) + "%")
    # print("PERSON ------> TP:" + str(P_TP) + " ---- FP:" + (str(PO_FP + PL_FP)))
    # print("LOCATION ------> TP:" + str(L_TP) + " ---- FP:" + (str(LO_FP + LP_FP)))


# Method to read and filter data from the dat files into an array
def read_dat_file(dat_writing):
    # count = 0
    for line in dat_writing:
        if line.strip() == '':  # if the line is empty
            # count += 1
            # print("---SKIPPED LINE " + str(count))
            continue

        l = line.split('(')

        # Remove the empty spaces in start and end, '\n' and '(' character
        try:
            l[0] = l[0].strip()
            l[1] = l[1].strip()
            l[1] = (l[1])[:-1]
            if 'GPE' in l[1]:  # Change GPE to LOCATION when storing into dat_array
                l[1] = 'LOCATION'
        except IndexError:  # Ignore the line if missing text or label
            continue

        dat_array.append(l)
        # count += 1
        # print("LINE " + str(count) + " EXTRACTED")


""" ------------------------------------ CODE EXECUTION BELOW ----------------------------------------- """

# Set path to your own path with all the dat and txt files in the same folder
datasetPath = 'C:/Users/LuoKe/OneDrive/Documents/EntireDataset/'
files = array(os.listdir(datasetPath))

fileCount = 0;
while True:
    if fileCount >= 554:
        break

    # Keeping reference to the current file and the next file
    currentFile = files[fileCount].split('.')
    if fileCount + 1 < 554:
        nextFile = files[fileCount + 1].split('.')

    if files[fileCount].endswith('.dat'):
        # Check if the next file is a .txt file with same student ID number
        if currentFile[0] == nextFile[0] and nextFile[1] == 'txt':
            print("------ Processing file: " + currentFile[0] + " ------ \r\n\r\n")

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
            chunk_label_data(txtWriting)
            ner_evaluation_and_comparison()

            dat_array.clear()
            results_array.clear()
            fileCount += 2

print_confusion_matrix()
print_FPR()