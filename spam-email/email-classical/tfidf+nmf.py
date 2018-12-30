import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

no_head_train_path_0 = 'PU/ham/'
no_head_train_path_1 = 'PU/spam/'

import os, re, string
import numpy as np

def clean_text(text):
    text = text.decode('utf-8')
    while '\n' in text:
        text = text.replace('\n', ' ')
    while '  ' in text:
        text = text.replace('  ', ' ')
    words = text.split()
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    stripped = []
    for token in words: 
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            stripped.append(new_token.lower())
    text = ' '.join(stripped)
    return text

def get_data(path):
    text_list = list()
    files = os.listdir(path)
    for text_file in files:
        file_path = os.path.join(path, text_file)
        read_file = open(file_path,'r+')
        read_text = read_file.read()
        read_file.close()
        cleaned_text = clean_text(read_text)
        text_list.append(cleaned_text)
    return text_list, files

no_head_train_0, temp = get_data(no_head_train_path_0)
no_head_train_1, temp1 = get_data(no_head_train_path_1)

no_head_train = no_head_train_0 + no_head_train_1
no_head_labels_train = ([0] * len(no_head_train_0)) + ([1] * len(no_head_train_1))

def vocabularymat(TEXTFILES,VOC,PLAY,METHOD):
    
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    if (METHOD == "TFIDF"):
        voc = TfidfVectorizer()
        voc.fit(VOC)
    
        if (PLAY == "TRAIN"):
            TrainMat = voc.transform(TEXTFILES)
            return TrainMat

        if (PLAY =="TEST"):
            TestMat = voc.transform(TEXTFILES)
            return TestMat
    
    if (METHOD == "TDM"):
        voc = CountVectorizer()
        voc.fit(VOC)
    
        if (PLAY == "TRAIN"):
            TrainMat = voc.transform(TEXTFILES)
            return TrainMat

        if (PLAY =="TEST"):
            TestMat = voc.transform(TEXTFILES)
            return TestMat

TrainMat = vocabularymat(no_head_train,no_head_train,PLAY= "TRAIN",METHOD="TFIDF")

data = TrainMat.todense()
datalabel = no_head_labels_train

Traindata = data

def Featurelearning(Data, Method):
    from sklearn.decomposition import TruncatedSVD, NMF
    if (Method == "SVD"):
        model = TruncatedSVD(n_components=30, n_iter=7, random_state=42)
        Matrix = model.fit_transform(Data)
    if (Method == "NMF"):
        model = NMF(n_components=30, init='random', random_state=0)
        Matrix = model.fit_transform(Data)
    return Matrix

X_train = Featurelearning(Traindata, Method="NMF")
y_train = datalabel

X_train, X_test, y_train, y_test = train_test_split(data, datalabel, test_size=0.33, random_state=42)

print("---------------------------------------------------------------")
unique, counts = np.unique(y_train, return_counts=True)
print(np.asarray((unique, counts)).T)

unique, counts = np.unique(y_test, return_counts=True)
print(np.asarray((unique, counts)).T)

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)

model = LogisticRegression()
model.fit(X_train, y_train)

# make predictions
expected = y_test
predicted = model.predict(X_test)
proba = model.predict_proba(X_test)
np.savetxt('res/predicted_LR_TFIDF+nmf.txt', proba)

accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted, average="binary")
precision = precision_score(expected, predicted , average="binary")
f1 = f1_score(expected, predicted , average="binary")
print(model)
print("Accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f-score")
print("%.3f" %f1)

cm = metrics.confusion_matrix(expected, predicted)
print(cm)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)

# make predictions
expected = y_test
predicted = model.predict(X_test)
proba = model.predict_proba(X_test)
np.savetxt('res/predicted_NB_TFIDF+nmf.txt', proba)

accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted, average="binary")
precision = precision_score(expected, predicted , average="binary")
f1 = f1_score(expected, predicted , average="binary")
print(model)
print("Accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f-score")
print("%.3f" %f1)

cm = metrics.confusion_matrix(expected, predicted)
print(cm)


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(X_train, y_train)

# make predictions
expected = y_test
predicted = model.predict(X_test)
proba = model.predict_proba(X_test)
np.savetxt('res/predicted_KNN_TFIDF+nmf', proba)

accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted, average="binary")
precision = precision_score(expected, predicted , average="binary")
f1 = f1_score(expected, predicted , average="binary")
print(model)
print("Accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f-score")
print("%.3f" %f1)

cm = metrics.confusion_matrix(expected, predicted)
print(cm)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# make predictions
expected = y_test
predicted = model.predict(X_test)
proba = model.predict_proba(X_test)
np.savetxt('res/predicted_DT_TFIDF+nmf.txt', proba)

accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted, average="binary")
precision = precision_score(expected, predicted , average="binary")
f1 = f1_score(expected, predicted , average="binary")
print(model)
print("Accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f-score")
print("%.3f" %f1)

cm = metrics.confusion_matrix(expected, predicted)
print(cm)

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=100)
model.fit(X_train, y_train)

# make predictions
expected = y_test
predicted = model.predict(X_test)
proba = model.predict_proba(X_test)
np.savetxt('res/predicted_AB_TFIDF+nmf.txt', proba)

accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted, average="binary")
precision = precision_score(expected, predicted , average="binary")
f1 = f1_score(expected, predicted , average="binary")
print(model)
print("Accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f-score")
print("%.3f" %f1)

cm = metrics.confusion_matrix(expected, predicted)
print(cm)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model = model.fit(X_train, y_train)

# make predictions
expected = y_test
predicted = model.predict(X_test)
proba = model.predict_proba(X_test)
np.savetxt('res/predicted_RF_TFIDF+nmf.txt', proba)

accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted, average="binary")
precision = precision_score(expected, predicted , average="binary")
f1 = f1_score(expected, predicted , average="binary")
print(model)
print("Accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f-score")
print("%.3f" %f1)

cm = metrics.confusion_matrix(expected, predicted)
print(cm)


from sklearn import svm
from sklearn.svm import SVC
model = svm.SVC(kernel='linear', C=1000,probability=True)
model.fit(X_train, y_train)

# make predictions
expected = y_test
predicted = model.predict(X_test)
proba = model.predict_proba(X_test)
np.savetxt('res/predicted_SVM-linear_TFIDF+nmf.txt', proba)

accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted, average="binary")
precision = precision_score(expected, predicted , average="binary")
f1 = f1_score(expected, predicted , average="binary")
print(model)
print("Accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f-score")
print("%.3f" %f1)

cm = metrics.confusion_matrix(expected, predicted)
print(cm)

from sklearn import svm
from sklearn.svm import SVC
model = svm.SVC(kernel='rbf', C=1000,probability=True)
model.fit(X_train, y_train)

# make predictions
expected = y_test
predicted = model.predict(X_test)
proba = model.predict_proba(X_test)
np.savetxt('res/predicted_SVM-rbf_TFIDF+nmf.txt', proba)

accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted, average="binary")
precision = precision_score(expected, predicted , average="binary")
f1 = f1_score(expected, predicted , average="binary")
print(model)
print("Accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f-score")
print("%.3f" %f1)

cm = metrics.confusion_matrix(expected, predicted)
print(cm)
