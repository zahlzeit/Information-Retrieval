import os
import random
import string
import pickle
import nltk
import numpy as np
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from collections import defaultdict
from nltk import FreqDist
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


stopwords = set(stopwords.words('english'))


def setup_doc():
    docs = [] #(label, text)
    with open('myTrain.csv', 'r', encoding='utf8') as datafile:
        for row in datafile:
            parts = row.split(',')
            doc = (parts[0], parts[1].strip())

            docs.append(doc)

    # print(docs)
    return docs


def clean_text(text):
    #remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))

    #convert to lower case
    text = text.lower()
    return text


def get_tokens(text):
    #get individual words
    tokens = word_tokenize(text)
    #remove useless common words
    tokens = [t for t in tokens if not t in stopwords]
    return tokens 


def print_frequency_dist(docs):
    tokens = defaultdict(list)

    #list of all words for each category
    for doc in docs:
        doc_label = doc[0]

        # doc_text = doc[1]
        doc_text = clean_text(doc[1])
        # doc_tokens = word_tokenize(doc_text)
        doc_tokens = get_tokens(doc_text)
        tokens[doc_label].extend(doc_tokens)


        # tokens[doc_label].extend(doc_tokens)

    for category_label, category_tokens in tokens.items():
        print(category_label)
        fd = FreqDist(category_tokens)
        print(fd.most_common(20))


def get_splits(docs):
    #scramble docs
    random.shuffle(docs)

    X_train = [] #training docs
    y_train = [] #training label

    X_test = [] #training docs
    y_test = [] #training label

    pivot = int(.8 * len(docs))
    # print(pivot)
    # print(len(docs))
    # print(docs[2000][0])
    
    for i in range(pivot):
        X_train.append(docs[i][1])
        
        X_test.append(docs[i][0])

    for i in range(pivot, len(docs)):
        y_train.append(docs[i][1])
        
        y_test.append(docs[i][0])

    return X_train, y_train, X_test, y_test


def evaluate_classifier(title, classifier, vectorizer, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_tfidf)

    precision = metrics.precision_score(y_test, y_pred, pos_label=1, average='micro')
    recall = metrics.recall_score(y_test, y_pred, pos_label=1, average='micro')
    f1 = metrics.f1_score(y_test, y_pred, pos_label=1, average='micro')

    print("%s \t %f \t %f \t %f \n" % (title, precision, recall, f1))


def train_classifier(docs):
    X_train, X_test, y_train, y_test = get_splits(docs)

    # object to turn text to vectors
    my_vectorizer = CountVectorizer(stop_words = 'english',
                                    ngram_range=(1, 3),
                                    min_df=3,
                                    analyzer='word')

    #create doc-term matrix
    dtm = my_vectorizer.fit_transform(X_train)

    # train Naive Bayes classifier
    naive_bayes_classifier = MultinomialNB().fit(dtm, y_train)

    evaluate_classifier("Naive Bayes\tTrain\t", naive_bayes_classifier, my_vectorizer, X_train, y_train)
    evaluate_classifier("Naive Bayes\tTest\t", naive_bayes_classifier, my_vectorizer, X_test, y_test)

    # #store the classifier 
    clf_filename = 'naive_bayes_classifer.pkl'
    pickle.dump(naive_bayes_classifier, open(clf_filename, 'wb'))

    # #store vectorizer to transform new data
    vec_filename = 'count_vectorizer.pkl'
    pickle.dump(my_vectorizer, open(vec_filename, 'wb'))


def classify(text):
    #load classifier
    clf_filename = 'naive_bayes_classifer.pkl'
    nb_clf = pickle.load(open(clf_filename, 'rb'))

    #vectorize the new text
    vec_filename = 'count_vectorizer.pkl'
    vectorizer = pickle.load(open(vec_filename, 'rb'))

    pred = nb_clf.predict(vectorizer.transform([text]))

    return pred[0] 


if __name__ == '__main__':
    # docs = setup_doc()
    # print_frequency_dist(docs)
    # print(docs)
    # train_classifier(docs)

    new_doc = "The study involved 579 children with ADHD who were part of longitudinal research that began to identify participants in the mid 1990’s. After setting the initial baseline, and after participants were provided treatment that included a variety of methods depending on their grouping over 14 months, they were then assessed at eight stages, ranging from two to sixteen years after their initial involvement. The method used included multiple questionnaires and interviews."
    classify(new_doc)

    print('Done')