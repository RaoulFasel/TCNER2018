import csv
import nltk
import keras
from sklearn.cross_validation import train_test_split

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

data = csv.DictReader(open("trainset.txt"), fieldnames=["id", "conference", "title"], delimiter='\t')
testdata = csv.DictReader(open("testset.txt"), fieldnames=["id", "title"], delimiter='\t')
vectorizer = CountVectorizer(min_df=0.002, max_df=0.5, stop_words=stop, token_pattern=r"\b[^\d\W]+\b",
                             strip_accents="ascii")
print(vectorizer.get_params())


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


def create_BC(x,y):
    clf = GaussianNB()
    clf.fit(x, y)

    return clf


def create_callbacks():
    return [keras.callbacks.EarlyStopping(monitor='val_acc',
                                          min_delta=0.01,
                                          patience=5,
                                          verbose=0, mode='auto'),
            ModelCheckpoint(filepath='temp/weights.hdf5', verbose=1, save_best_only=True)
            ]

def write_predictions(name, result):
    output = open("output/"+name+".txt", 'w')
    id = 0
    for r in result:
        output.write(str(id) + "\t" + r + "\n")
        id += 1
    output.close()


le = preprocessing.LabelEncoder()
test = create_test_set(testdata)
X, Y_text = create_train_set(data)
X = vectorizer.fit_transform(X)
Y = le.fit_transform(Y_text)
Y = categorical_labels = to_categorical(Y, num_classes=None)
X_train, X_test, y_train, y_test = train_test_split(X.toarray(), Y_text, test_size=0.33, random_state=42)

#neural network model
DNNmodel = createDNN(vectorizer.get_feature_names(), le.classes_)

DNNmodel.fit(X.toarray(), Y, epochs=150, batch_size=300, validation_split=0.2, shuffle=True,
             callbacks=create_callbacks())

result = DNNmodel.predict(vectorizer.transform(test).toarray())
write_predictions("DNN",[le.classes_[r.tolist().index(max(r))] for r in result])

#bayes model

bayesmodel = create_BC(X_train,y_train)
score = bayesmodel.score(X_test, y_test)
result = bayesmodel.predict(vectorizer.transform(test).toarray())
write_predictions("bayes",result)