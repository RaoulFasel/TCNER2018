import csv
import nltk
import keras

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from nltk.corpus import stopwords
from keras.utils.np_utils import to_categorical

nltk.download('punkt')
nltk.download('stopwords')

stop = set(stopwords.words('english'))

tokenizer = nltk.RegexpTokenizer(r'\w+')

data = csv.DictReader(open("trainset.txt"), fieldnames=["id", "conference", "title"], delimiter='\t')
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


def createDNN(features, labels):
    model = Sequential()
    model.add(Dense(150, input_dim=len(features), activation='relu'))
    model.add(Dense(15, activation='tanh'))
    model.add(Dense(len(labels), activation='sigmoid'))
    return model


le = preprocessing.LabelEncoder()
X, Y = create_train_set(data)
X = vectorizer.fit_transform(X)
Y = le.fit_transform(Y)
Y = categorical_labels = to_categorical(Y, num_classes=None)
print(vectorizer.get_feature_names())
print(len(vectorizer.get_feature_names()))
print(len(le.classes_))
print(Y[5])
print(X.toarray().shape)
model = createDNN(vectorizer.get_feature_names(), le.classes_)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0.01,
                              patience=5,
                              verbose=0, mode='auto')
model.fit(X.toarray(), Y, epochs=150, batch_size=10, validation_split=0.2, shuffle=True)
