# COMP700 TEXT AND VISION INTELLIGENCE - ASSIGNMENT 1
# Kelly Luo (17985065)

from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import nltk
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

path = 'C:\\Users\\LuoKe\\OneDrive\\Documents\\17985065_1.txt'
text  = open(path)
# print(text.read())

txtWriting = open(path, 'r').read()

path2 = 'C:\\Users\\LuoKe\\OneDrive\\Documents\\17985065_1.dat'
dat  = open(path2)
print(dat.read())

datWriting = open(path2, 'r').read()
print('\r\n')

# txt = "Jacinda Ardern is the Prime Minister of New Zealand but Roenzo isn't."
# print (get_continuous_chunks(txt))

dat_array = []
with open(path2) as datFile:
    dat_array = datFile.readlines()

# count = 0
# while True:
#     count += 1
#
#     # Get next line from file
#     line = datWriting.readline()
#
#     # if line is empty
#     # end of file is reached
#     if not line:
#         break
#     print("Line{}: {}".format(count, line.strip()))

# Passing to analyse the article text and chunk with labels



for sent in nltk.sent_tokenize(txtWriting): #for all the words in the text article
   for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))): #for all the connected chunks
      if hasattr(chunk, 'label'):

         print(chunk.label(), ' '.join(c[0] for c in chunk))


# import nltk
# import sklearn
# from sklearn.metrics import confusion_matrix
# from collections import defaultdict
# refsets = defaultdict(set)
# testsets = defaultdict(set)
# labels = []
# tests = []
# for i, (feats, label) in enumerate(testsets):
#     refsets[label].add(i)
#     observed = nltk.classify(feats)
#     testsets[observed].add(i)
#     labels.append(label)
#     tests.append(observed)
#
# print(sklearn.metrics.confusion_matrix(labels, tests))
# print(nltk.ConfusionMatrix(labels, tests))