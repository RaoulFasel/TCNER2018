import csv
import nltk
import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense

from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

stop = set(stopwords.words('english'))

tokenizer = nltk.RegexpTokenizer(r'\w+')

data = csv.DictReader(open("trainset.txt"), fieldnames=["id", "conference", "title"], delimiter='\t')
vectorizer = CountVectorizer(min_df=2,max_df=100,stop_words=stop,token_pattern=r'\w+',analyzer="word")

# code that is not needed because of amazing sklearn
# def preprocessor(data, tokenizer, min_count, max_count, min_length):
#     conferences = []
#     wordcount = collections.Counter()
#     for paper in data:
#         conference = paper['conference']
#         title = [i for i in tokenizer.tokenize(paper['title'].lower()) if i not in stop]
#         if conference not in conferences:
#             conferences.append(conference)
#         for word in title:
#             wordcount[word] += 1
#     wordcount = collections.Counter(
#         {k: wordcount[k] for k in wordcount if
#          (wordcount[k] >= min_count and wordcount[k] <= max_count and len(k) >= min_length)})
#
#     return conferences, wordcount
#     print(wordcount.most_common())
#     print(len(wordcount.most_common()[::-1]))
#     print(conferences[0:10])
#     print(len(conferences))


def create_train_set(data):
    X = []
    Y = []
    for paper in data:
        Y.append(paper['conference'])
        X.append(paper['title'])
    return X, Y


def createDNN(features,labels):
    model = Sequential()
    model.add(Dense(150, input_dim=len(features), activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

le = preprocessing.LabelEncoder()
X,Y = create_train_set(data)
X = vectorizer.fit_transform(X)
Y = le.fit_transform(Y)
print(vectorizer.get_feature_names())
print(len(vectorizer.get_feature_names()))
print(len(le.classes_))
print(Y[5])
print(X.toarray().shape)
model = createDNN(vectorizer.get_feature_names(),list(le.classes_))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X.toarray(), Y, epochs=150, batch_size=10)