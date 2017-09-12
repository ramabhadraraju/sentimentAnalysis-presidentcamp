import numpy as np  # a conventional alias

from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import os

os.chdir("C:\\UTA Courses\\spring\\Data Science 5378\\textminingproject\\tweets\\text") #This is where my files are stored
files = glob.glob("*.csv") #get text file names
 
corpus = []

for f in files:
    f_input = open(f)
    txtlist = f_input.read().splitlines()
    #txt = [w.strip() for w in txt.split()]
    for item in txtlist:    
        corpus.append(item)
    f_input.close()
vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 1)
dtm = vectorizer.fit_transform(corpus)
terms= np.array(vectorizer.get_feature_names())
dtm.shape
len(terms)
from sklearn import decomposition
num_topics = 25
num_top_words = 20
clf = decomposition.NMF(n_components = num_topics, random_state=1)
doctopic = clf.fit_transform(dtm)
topic_words = []
for topic in clf.components_:
    word_idx = np.argsort(topic)[::-1][0:num_top_words]
    topic_words.append([terms[i] for i in word_idx])
for t in range(len(topic_words)):
    print("Topic {}: {}".format(t, ' '.join(topic_words[t][:21])))

