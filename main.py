import csv
import nltk
import keras
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout
from nltk.corpus import stopwords
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.naive_bayes import GaussianNB

nltk.download('stopwords')

stop = set(stopwords.words('english'))


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


def create_BC(x, y):
    clf = GaussianNB()
    clf.fit(x, y)

    return clf


def create_callbacks(name):
    return [keras.callbacks.EarlyStopping(monitor='val_acc',
                                          min_delta=0.01,
                                          patience=5,
                                          verbose=0, mode='auto'),
            ModelCheckpoint(filepath='temp/'+name+'weights.hdf5', verbose=1, save_best_only=True)
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

    for clf in clfs:
        name = clf[3]
        params = clf[5]
        le = None
        y=Y
        y_test=Y_test
        if(clf[2]):
            le = preprocessing.LabelEncoder()
            y = convert_to_cat(y,le)
            y_test = convert_to_cat(y_test,le)
        pre = clf[1](**clf[4])
        x = pre.fit_transform(X).toarray()
        x_test = pre.transform(X_test).toarray()
        x_submit = pre.transform(X_submit).toarray()
        c = clf[0](pre.get_feature_names(), le.classes_)
        print(pre.get_params())
        do_classifier(c,x,y,x_test,y_test,x_submit,le,params,name)


def do_classifier(clf, x, y, x_test, y_test, x_submit, le,params,name):
    params["callbacks"] = create_callbacks(name)
    clf.fit(x,y,**params)
    print(clf.evaluate(x_test,y_test))
    predict = clf.predict(x_test)
    print(confusion_matrix([le.classes_[r.tolist().index(max(r))] for r in y_test],
                           [le.classes_[r.tolist().index(max(r))] for r in predict], labels=le.classes_))
    submit_predict = clf.predict(x_submit)
    write_predictions(name+"txt", [le.classes_[r.tolist().index(max(r))] for r in submit_predict])


classifiers = [
    [createDNN, CountVectorizer, True,"DNN1",{"min_df":0.001, "max_df":0.5, "stop_words":stop, "token_pattern":r"\b[^\d\W]+\b","strip_accents":"ascii"},{"epochs":150, "batch_size":300, "validation_split":0.2, "shuffle":True,"callbacks":None, "verbose":0}],
    [createDNN, CountVectorizer, True,"DNN2",{"min_df": 0.0008, "max_df": 0.5, "stop_words": stop, "token_pattern": r"\b[^\d\W]+\b","strip_accents": "ascii"},{"epochs":150, "batch_size":300, "validation_split":0.2, "shuffle":True,"callbacks":None, "verbose":0}],
    [createDNN, CountVectorizer, True,"DNN3",{"min_df": 0.0007, "max_df": 0.5, "stop_words": stop, "token_pattern": r"\b[^\d\W]+\b", "strip_accents": "ascii"},{"epochs":150, "batch_size":300, "validation_split":0.2, "shuffle":True,"callbacks":None, "verbose":0}]
]
do_all_classifiers(classifiers)

# bayes model
# X_train, X_test, y_train, y_test = train_test_split(X.toarray(), Y_text, test_size=0.33, random_state=42)
# bayesmodel = create_BC(X_train, y_train)
# score = bayesmodel.score(X_test, y_test)
# result = bayesmodel.predict(vectorizer.transform(test).toarray())
# write_predictions("bayes", result)
