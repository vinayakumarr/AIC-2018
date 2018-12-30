from __future__ import print_function
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import roc_curve, auc
import itertools
from scipy import interp
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error, roc_curve, classification_report,auc)



te = pd.read_csv('dataset1/testlabel.csv', header=None)
te1 = pd.read_csv('dataset1/bigram.txt', header=None)
te2 = pd.read_csv('dataset1/cnn.txt', header=None)
te3 = pd.read_csv('dataset1/cnn-lstm.txt', header=None)




C = te.iloc[:,0]
C1 = te1.iloc[:,0]
C2 = te2.iloc[:,0]
C3 = te3.iloc[:,0]





c = np.array(C)
c1 = np.array(C1)
c2 = np.array(C2)
c3 = np.array(C3)




def calc_macro_roc(fpr, tpr):
    """Calcs macro ROC on log scale"""
    # Create log scale domain
    all_fpr = sorted(itertools.chain(*fpr))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(tpr)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    return all_fpr, mean_tpr / len(tpr), auc(all_fpr, mean_tpr) / len(tpr)



t_fpr, t_tpr, _ = roc_curve(c, c2)
fpr1 = []
tpr1 = []

fpr1.append(t_fpr)
tpr1.append(t_tpr)

cnn_binary_fpr, cnn_binary_tpr, cnn_binary_auc = calc_macro_roc(fpr1, tpr1)



t_fpr, t_tpr, _ = roc_curve(c, c3)
fpr2 = []
tpr2 = []

fpr2.append(t_fpr)
tpr2.append(t_tpr)

cnnlstm_binary_fpr, cnnlstm_binary_tpr, cnnlstm_binary_auc = calc_macro_roc(fpr2, tpr2)



from matplotlib import pyplot as plt



plt.figure(figsize=(5,4))
SMALL_SIZE = 9
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)



plt.plot(cnn_binary_fpr, cnn_binary_tpr,label='cnn (AUC = %.4f)' % (cnn_binary_auc, ), rasterized=True,linewidth=0.5)
plt.plot(cnnlstm_binary_fpr, cnnlstm_binary_tpr,label='cnn-lstm (AUC = %.4f)' % (cnnlstm_binary_auc, ), rasterized=True,linewidth=1)


plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(linestyle='dotted')


plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('ROC - Binary Classification', fontsize=10)
plt.legend(loc="lower right", prop={'size':9})
plt.tick_params(axis='both', labelsize=9)
plt.show()

#plt.savefig("ROC.png",dpi=110)



