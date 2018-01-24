import csv
import nltk
import collections

import keras

from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

stop = set(stopwords.words('english'))

tokenizer = nltk.RegexpTokenizer(r'\w+')

reader = csv.DictReader(open("trainset.txt"), fieldnames=["id", "conference", "title"], delimiter='\t')


def preprocessor(data, tokenizer, min_count, max_count, min_length):
    conferences = []
    wordcount = collections.Counter()
    for paper in reader:
        conference = paper['conference']
        title = [i for i in tokenizer.tokenize(paper['title'].lower()) if i not in stop]
        if conference not in conferences:
            conferences.append(conference)
        for word in title:
            wordcount[word] += 1
    wordcount = collections.Counter(
        {k: wordcount[k] for k in wordcount if (wordcount[k] >= min_count and wordcount[k] <= max_count and len(k) >= min_length)})
    print(wordcount.most_common())
    print(len(wordcount.most_common()[::-1]))
    print(conferences[0:10])
    print(len(conferences))


preprocessor(reader, tokenizer, 2, 100 , 2)
