import csv
import nltk
import keras
import numpy
from keras.optimizers import Adam
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn import preprocessing, svm
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Embedding, Convolution1D, GlobalMaxPooling1D, Conv2D, MaxPool2D, \
    Flatten, Concatenate, Reshape
from nltk.corpus import stopwords
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.naive_bayes import GaussianNB
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from tabulate import tabulate
from nltk.tokenize import RegexpTokenizer
import string
from data_helpers import *
import csv
from operator import itemgetter
import numpy as np
import re
import itertools
from collections import Counter

from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from nltk import RegexpTokenizer

regex = RegexpTokenizer(r"\b[^\d\W]+\b")
nltk.download('stopwords')
nltk.download('wordnet')
stop = set(stopwords.words('english'))

class Reg(object):
    def __init__(self):
        pass
    def __call__(self, doc):
        return [t for t in regex.tokenize(doc) if t not in stop]
    def __str__(self):
        return "Regular expression"


class RegWithStop(object):
    def __init__(self):
        pass
    def __call__(self, doc):
        return [t for t in regex.tokenize(doc)]
    def __str__(self):
        return "Regular expression with stopwords "


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in regex.tokenize(doc) if t not in stop]
    def __str__(self):
        return "Lemmatization"


class StemTokenizer(object):
    def __init__(self):
        self.wnl = nltk.SnowballStemmer("english", ignore_stopwords=True)

    def __call__(self, doc):
        return [self.wnl.stem(t) for t in regex.tokenize(doc)]
    def __str__(self):
        return "Stemmning"

class LemmaTokenizerWithStop(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in regex.tokenize(doc)]
    def __str__(self):
        return "Lemmatization with stopwords"


class StemTokenizerWithStop(object):
    def __init__(self):
        self.wnl = nltk.SnowballStemmer("english", ignore_stopwords=False)

    def __call__(self, doc):
        return [self.wnl.stem(t) for t in regex.tokenize(doc)]
    def __str__(self):
        return "Stemmning with stopwords"


def create_train_set(data):
    x = []
    y = []
    for paper in data:
        y.append(paper['conference'])
        x.append(paper['title'])
    return x, y


def create_test_set(data):
    x = []
    for paper in data:
        x.append(paper['title'])
    return x


def createDNN(features, labels):
    m = Sequential()
    m.add(Dense(150, input_dim=len(features), activation='tanh'))
    m.add(Dropout(rate=.1))
    m.add(Dense(15, activation='tanh'))
    m.add(Dense(len(labels), activation='softmax'))
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return m

def createCNN(sequence_length,vocabulary_size,embedding_dim,filter_sizes,num_filters,drop,output_length):
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=output_length, activation='softmax')(dropout)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def create_BC():
    clf = GaussianNB()
    return clf


def create_svm():
    clf = svm.SVC(decision_function_shape='ovo')
    return clf


def create_callbacks(name):
    return [keras.callbacks.EarlyStopping(monitor='val_acc',
                                          min_delta=0.01,
                                          patience=5,
                                          verbose=0, mode='auto'),
            ModelCheckpoint(filepath='temp/' + name + '_best_weights.hdf5', verbose=1, save_best_only=True)
            ]


def write_predictions(name, result):
    output = open("output/" + name + ".txt", 'w')
    id = 0
    for r in result:
        output.write(str(id) + "\t" + r + "\n")
        id += 1
    output.close()


def convert_to_cat(Y, le):
    Y = le.fit_transform(Y)
    Y = to_categorical(Y, num_classes=None)
    return Y


def do_all_classifiers(clfs):
    data = csv.DictReader(open("trainset.txt"), fieldnames=["id", "conference", "title"], delimiter='\t')
    submitdata = csv.DictReader(open("testset.txt"), fieldnames=["id", "title"], delimiter='\t')
    X, Y = create_train_set(data)
    X_submit = create_test_set(submitdata)

    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    run_results = []
    for clf in clfs:
        print(clf)
        name = clf[3]
        params = clf[5]
        settings = clf[4]
        le = None
        c = None
        y = Y
        y_test = Y_test
        if(clf[1] ==CountVectorizer):
            pre = clf[1](**clf[4])
            x = pre.fit_transform(X).toarray()
            n_features =len(pre.get_feature_names())
            x_test = pre.transform(X_test).toarray()
            x_submit = pre.transform(X_submit).toarray()
        if clf[0]==createCNN:

            label_encoder = LabelEncoder()


            x, y, vocabulary, vocabulary_inv = load_data(X,Y,settings['tokenizer'])

            y_test = np.array(y_test)
            y = label_encoder.fit_transform(y)
            y = to_categorical(y)
            y_test = label_encoder.transform(y_test)
            y_test = to_categorical(y_test)
            sequence_length = x.shape[1]  # 56
            output_length = y.shape[1]
            vocabulary_size = len(vocabulary_inv)  # 18765
            x_test = load_test_data(X_test,vocabulary,settings['tokenizer'],sequence_length)
            x_submit = load_test_data(X_submit,vocabulary,settings['tokenizer'],sequence_length)
            embedding_dim = 256
            filter_sizes = [3, 4, 5]
            num_filters = 512
            drop = 0.5
            c = createCNN(sequence_length,vocabulary_size,embedding_dim,filter_sizes,num_filters,drop,output_length)
            le = label_encoder
            n_features = vocabulary_size
        if (clf[2]):
            # check for label encode)r(needed for DNN not needed for CNN)
            le = preprocessing.LabelEncoder()
            y = convert_to_cat(y,le)
            y_test = convert_to_cat(y_test,le)
            c = clf[0](pre.get_feature_names(), le.classes_)

        if clf[0]==create_BC:
            c = clf[0]()
        run_results.append(do_classifier(c, x, y, x_test, y_test, x_submit, le, params, name, settings,n_features))
    write_results(run_results)


def write_results(results):
    file = open("output/results.txt", 'w')
    for r in results:
        file.write(r[0] + '\n')
        file.write("accuracy = " + str(r[2]) + "\n")
        file.write(tabulate(r[1], tablefmt='latex', headers=r[4]) + '\n')
    file.write(tabulate(sorted([[i[0], i[2], i[3]] for i in results],key=itemgetter(2)), headers=["Name", "Accuracy", "F1"], tablefmt='latex'))
    file.write(tabulate([[i[0],i[5]['min_df'],i[5]['max_df'],i[5]['tokenizer'],i[6]] for i in results], headers=["min_df", "max_df", "Tokenizer","Features"], tablefmt='latex'))

def do_classifier(clf, x, y, x_test, y_test, x_submit, le, params, name,settings,n_features):
    if params:
        # check for extra classifier parameters


        params["callbacks"] = create_callbacks(name)
        clf.fit(x, y, **params)
    else:
        clf.fit(x, y)
    # print(clf.evaluate(x_test,y_test))
    predict = clf.predict(x_test)
    submit_predict = clf.predict(x_submit)
    if le:
        # check for label encoder(needed for DNN)
        y_test, predict = ([le.classes_[r.tolist().index(max(r))] for r in y_test],
                           [le.classes_[r.tolist().index(max(r))] for r in predict])
        print(confusion_matrix(y_test, predict, labels=le.classes_))
        write_predictions(name + "txt", [le.classes_[r.tolist().index(max(r))] for r in submit_predict])
        return [name, confusion_matrix(y_test, predict, labels=le.classes_), accuracy_score(y_test, predict),
                f1_score(y_test, predict, average='micro'), le.classes_,settings,n_features]
    else:
        print(confusion_matrix(y_test, predict))
        write_predictions(name + "txt", submit_predict)
        return [name, confusion_matrix(y_test, predict, labels=['INFOCOM', 'ISCAS', 'SIGGRAPH', 'VLDB', 'WWW']),
                accuracy_score(y_test, predict), f1_score(y_test, predict, average='micro'),
                ['INFOCOM', 'ISCAS', 'SIGGRAPH', 'VLDB', 'WWW'],settings,n_features]


# all classifiers to be trained and evaluated
# Format:
# [  Classifier,Vectorizer,Label to catogeries, name, Vectorizer parameters(dict) , classifier parameters(dict)  ]
classifiers = [
    [createCNN, None, False, "CNN1", {"tokenizer": StemTokenizerWithStop(), "min_df": "n/a", "max_df": "n/a",},
     {"epochs": 150, "batch_size": 32, "validation_split": 0.2, "shuffle": True, "callbacks": None, "verbose": 1}],

    [createCNN, None, False, "CNN2", {"tokenizer": LemmaTokenizerWithStop(), "min_df": "n/a", "max_df": "n/a",},
     {"epochs": 150, "batch_size": 32, "validation_split": 0.2, "shuffle": True, "callbacks": None, "verbose": 1}],
    [createCNN, None, False, "CNN3", {"tokenizer": RegWithStop(), "min_df": "n/a", "max_df": "n/a",},
     {"epochs": 150, "batch_size": 32, "validation_split": 0.2, "shuffle": True, "callbacks": None, "verbose": 1}],
    [createCNN, None, False, "CNN4", {"tokenizer": Reg(), "min_df": "n/a", "max_df":"n/a",},
     {"epochs": 150, "batch_size": 32, "validation_split": 0.2, "shuffle": True, "callbacks": None, "verbose": 1}],

    [create_BC, CountVectorizer, False, "NaiveBayes1",
     {"tokenizer": StemTokenizer(), "min_df": 0.0001, "max_df": 0.5, "stop_words": stop,
      "token_pattern": r"\b[^\d\W]+\b", "strip_accents": "ascii"},
     None],
    [create_BC, CountVectorizer, False, "NaiveBayes2",
     {"tokenizer": Reg(), "min_df": 0.0001, "max_df": 0.5, "stop_words": stop,
      "token_pattern": r"\b[^\d\W]+\b", "strip_accents": "ascii"},
     None],
    [create_BC, CountVectorizer, False, "NaiveBayes3",
     {"tokenizer": LemmaTokenizer(), "min_df": 0.0001, "max_df": 0.5, "stop_words": stop,
      "token_pattern": r"\b[^\d\W]+\b", "strip_accents": "ascii"},
     None],

    [createDNN, CountVectorizer, True, "DNN1",
     {"tokenizer": StemTokenizer(), "min_df": 0.0001, "max_df": 0.5, "stop_words": stop,
      "token_pattern": r"\b[^\d\W]+\b", "strip_accents": "ascii"},
     {"epochs": 150, "batch_size": 300, "validation_split": 0.2, "shuffle": True, "callbacks": None, "verbose": 0}],
    [createDNN, CountVectorizer, True, "DNN2",
     {"tokenizer": LemmaTokenizer(), "min_df": 0.0001, "max_df": 0.5, "stop_words": stop,
      "token_pattern": r"\b[^\d\W]+\b", "strip_accents": "ascii"},
     {"epochs": 150, "batch_size": 300, "validation_split": 0.2, "shuffle": True, "callbacks": None, "verbose": 0}],
    [createDNN, CountVectorizer, True, "DNN3",
     {"tokenizer": Reg(), "min_df": 0.0001, "max_df": 0.5, "stop_words": stop,
      "token_pattern": r"\b[^\d\W]+\b", "strip_accents": "ascii"},
     {"epochs": 150, "batch_size": 300, "validation_split": 0.2, "shuffle": True, "callbacks": None, "verbose": 0}],

    [createDNN, CountVectorizer, True, "DNN4",
     {"tokenizer": StemTokenizer(), "min_df": 0.00001, "max_df": 0.6, "stop_words": stop,
      "token_pattern": r"\b[^\d\W]+\b", "strip_accents": "ascii"},
     {"epochs": 150, "batch_size": 300, "validation_split": 0.2, "shuffle": True, "callbacks": None, "verbose": 0}],
    [createDNN, CountVectorizer, True, "DNN5",
     {"tokenizer": LemmaTokenizer(), "min_df": 0.00001, "max_df": 0.6, "stop_words": stop,
      "token_pattern": r"\b[^\d\W]+\b", "strip_accents": "ascii"},
     {"epochs": 150, "batch_size": 300, "validation_split": 0.2, "shuffle": True, "callbacks": None, "verbose": 0}],

    [createDNN, CountVectorizer, True, "DNN6",
     {"tokenizer": Reg(), "min_df": 0.00001, "max_df": 0.6, "stop_words": stop,
      "token_pattern": r"\b[^\d\W]+\b", "strip_accents": "ascii"},
     {"epochs": 150, "batch_size": 300, "validation_split": 0.2, "shuffle": True, "callbacks": None, "verbose": 0}],

    [create_svm, CountVectorizer, False, "SVM1",
     {"tokenizer": Reg(),"min_df": 0.0001, "max_df": 0.5, "stop_words": stop, "token_pattern": r"\b[^\d\W]+\b", "strip_accents": "ascii"},
     None]


    # [create_BC, CountVectorizer, False, "NaiveBayes1",
    #  { "tokenizer": None ,"min_df": 0.0007, "max_df": 0.5, "stop_words": stop,"token_pattern": r"\b[^\d\W]+\b", "strip_accents": "ascii"},
    #  None],
    # [create_BC, CountVectorizer, False, "NaiveBayes2",
    #  {"tokenizer": LemmaTokenizer(), "min_df": 0.0008, "max_df": 0.5, "stop_words": stop,
    #   "token_pattern": r"\b[^\d\W]+\b", "strip_accents": "ascii"}, None],
    # [createDNN, CountVectorizer, True, "DNN2",
    #  {"tokenizer": None ,"min_df": 0.0007, "max_df": 0.5, "stop_words": stop, "token_pattern": r"\b[^\d\W]+\b", "strip_accents": "ascii"},
    #  {"epochs": 150, "batch_size": 300, "validation_split": 0.2, "shuffle": True, "callbacks": None, "verbose": 0}],
    # [createDNN, CountVectorizer, True,"DNN3",{"tokenizer":None,"min_df": 0.00001, "max_df": 0.6, "stop_words": stop, "token_pattern": r"\b[^\d\W]+\b", "strip_accents": "ascii"},{"epochs":150, "batch_size":300, "validation_split":0.2, "shuffle":True,"callbacks":None, "verbose":0}]

]
print(len(classifiers))
do_all_classifiers(classifiers)
