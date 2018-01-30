import csv
import nltk
import keras
import numpy
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing, svm
from keras.models import Sequential
from keras.layers import Dense, Dropout
from nltk.corpus import stopwords
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.naive_bayes import GaussianNB
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from tabulate import tabulate
from nltk.tokenize import RegexpTokenizer
import string

regex = RegexpTokenizer(r"\b[^\d\W]+\b")
nltk.download('stopwords')
nltk.download('wordnet')
stop = set(stopwords.words('english'))

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in regex.tokenize(doc) if t not in stop]

class StemTokenizer(object):
    def __init__(self):
        self.wnl = nltk.SnowballStemmer("english", ignore_stopwords=True)
    def __call__(self, doc):
        return [self.wnl.stem(t) for t in regex.tokenize(doc)]

def remove_punc_deci(string):


    return


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
    m.add(Dense(150, input_dim=len(features), activation='relu'))
    m.add(Dropout(rate=.1))
    m.add(Dense(15, activation='tanh'))
    m.add(Dense(len(labels), activation='sigmoid'))
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return m


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
            ModelCheckpoint(filepath='temp/'+name+'_best_weights.hdf5', verbose=1, save_best_only=True)
            ]


def write_predictions(name, result):
    output = open("output/" + name + ".txt", 'w')
    id = 0
    for r in result:
        output.write(str(id) + "\t" + r + "\n")
        id += 1
    output.close()


def convert_to_cat(Y,le):
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
        name = clf[3]
        params = clf[5]
        le = None
        y=Y
        y_test=Y_test
        pre = clf[1](**clf[4])
        x = pre.fit_transform(X).toarray()
        print(pre.get_feature_names())
        x_test = pre.transform(X_test).toarray()
        x_submit = pre.transform(X_submit).toarray()
        print(pre.get_params())



        if(clf[2]):
            #check for label encode)r(needed for DNN)
            le = preprocessing.LabelEncoder()
            y = convert_to_cat(y,le)
            y_test = convert_to_cat(y_test,le)
            c = clf[0](pre.get_feature_names(), le.classes_)
        else:
            c = clf[0]()

        run_results.append(do_classifier(c,x,y,x_test,y_test,x_submit,le,params,name))
    write_results(run_results)

def write_results(results):
    file = open("output/results.txt",'w')
    for r in results:
        file.write(r[0]+'\n')
        file.write("accuracy = "+str(r[2])+"\n")
        file.write(tabulate(r[1],tablefmt='latex',headers=r[4])+'\n')
    file.write(tabulate([[i[0],i[2],i[3]] for i in results],headers=["Name","Accuracy","F1"],tablefmt='latex'))

def do_classifier(clf, x, y, x_test, y_test, x_submit, le,params,name):
    if params:
        # check for extra classifier parameters

        params["callbacks"] = create_callbacks(name)
        clf.fit(x,y,**params)
    else:
        clf.fit(x,y)
    #print(clf.evaluate(x_test,y_test))
    predict = clf.predict(x_test)
    submit_predict = clf.predict(x_submit)
    if le:
        print(le.classes_)
        # check for label encoder(needed for DNN)
        y_test,predict = ([le.classes_[r.tolist().index(max(r))] for r in y_test], [le.classes_[r.tolist().index(max(r))] for r in predict])
        print(confusion_matrix(y_test,predict, labels=le.classes_))
        return [name,confusion_matrix(y_test,predict, labels=le.classes_),accuracy_score(y_test,predict),f1_score(y_test,predict,average='micro'),le.classes_]
        write_predictions(name+"txt", [le.classes_[r.tolist().index(max(r))] for r in submit_predict])
    else:
        print(confusion_matrix(y_test,predict))
        return [name,confusion_matrix(y_test,predict,labels=['INFOCOM','ISCAS','SIGGRAPH','VLDB','WWW']),accuracy_score(y_test,predict),f1_score(y_test,predict,average='micro'),['INFOCOM','ISCAS','SIGGRAPH','VLDB','WWW']]
        write_predictions(name + "txt", submit_predict)




# all classifiers to be trained and evaluated
# Format:
# [  Classifier,Vectorizer,Label to catogeries, name, Vectorizer parameters(dict) , classifier parameters(dict)  ]
classifiers = [
    [createDNN, CountVectorizer, True,"DNN1",{"tokenizer":StemTokenizer(),"min_df":0.001, "max_df":0.5, "stop_words":stop, "token_pattern":r"\b[^\d\W]+\b","strip_accents":"ascii"},{"epochs":150, "batch_size":300, "validation_split":0.2, "shuffle":True,"callbacks":None, "verbose":0}],
    [create_BC, CountVectorizer, False, "NaiveBayes1",
     {"min_df": 0.0007, "max_df": 0.5, "stop_words": stop, "token_pattern": r"\b[^\d\W]+\b", "strip_accents": "ascii"},
     None],

    [create_BC, CountVectorizer, False,"NaiveBayes2",{"tokenizer":LemmaTokenizer(),"min_df": 0.0008, "max_df": 0.5, "stop_words": stop, "token_pattern": r"\b[^\d\W]+\b","strip_accents": "ascii"},None],
    [createDNN, CountVectorizer, True,"DNN2",{"min_df": 0.0007, "max_df": 0.5, "stop_words": stop, "token_pattern": r"\b[^\d\W]+\b", "strip_accents": "ascii"},{"epochs":150, "batch_size":300, "validation_split":0.2, "shuffle":True,"callbacks":None, "verbose":0}]
]
do_all_classifiers(classifiers)
