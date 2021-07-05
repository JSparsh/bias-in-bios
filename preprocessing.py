import nltk
# from nltk.corpus import stopwords
import tensorflow as tf
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from config import WORD2VEC_PATH, DATASET_NAMES, MASKED
# from gensim.models import Word2Vec
from gensim.models import KeyedVectors, Word2Vec
from joblib import dump, load
import numpy as np
import pandas as pd
import tensorflow_hub as hub
from config import CLASS_GROUP
from sampling import get_data_from_mongo
from sklearn.svm import SVC
from config import DATASET_NAMES, MASKED, SEED
import numpy as np
 
# nltk.download('punkt')
# nltk.download('stopwords')
stop_words=set(nltk.corpus.stopwords.words("english"))

def embedding_fit_transform(x_list, embedding, class_group, sampling, test_size, masking):
    print("processing preprocessing.embedding_fit_transform ...")
    
    if embedding == 'cv': 
        trained_embedding, X = count_vectorize_fit_transform(x_list)
    elif embedding == 'w2v':
        trained_embedding, X = pretrained_word2vec_fit_transform(x_list)
    elif embedding == 'self_w2v':
        trained_embedding, X = selftrained_word2vec_fit_transform(x_list)

    dump(trained_embedding,DATASET_NAMES['embedding',embedding,class_group,sampling,test_size,MASKED[masking]]+'.joblib')
    print("\t saving file :",DATASET_NAMES['embedding',embedding,class_group,sampling,test_size,MASKED[masking]])

    return(X)


def embedding_transform(x_list, embedding, class_group, sampling, test_size, masking ):
    # print("processing preprocessing.embedding_transform ...")
    
    trained_embedding = load(DATASET_NAMES['embedding',embedding,class_group,sampling,test_size,MASKED[masking]]+'.joblib')
    
    if embedding == 'cv': 
        X = count_vectorize_transform(trained_embedding,x_list)
    elif embedding == 'w2v'or embedding == 'self_w2v':
        X = word2vec_transform(trained_embedding,x_list)
    return(X)


''' Add this with every embedding

    -> RegexpTokenizer
    -> nltk english stop_words
    -> SnowballStemmer
'''
def preProcessAndTokenize(sentence):
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tokenized_words=tokenizer.tokenize(sentence.lower())
    filtered_words=[]
    for w in tokenized_words:
       if w.lower() not in stop_words:
           filtered_words.append(w)
    snowball_stemmer = SnowballStemmer('english', ignore_stopwords=True)   
    stemmed = [snowball_stemmer.stem(word) for word in filtered_words]
    return stemmed


def count_vectorize_fit_transform(x_list):
    vectorizer_binary = CountVectorizer(lowercase=True, preprocessor=None, binary=True, stop_words=None, tokenizer=preProcessAndTokenize)
    X = vectorizer_binary.fit_transform(x_list)
    print("         count vectorized with dimension : ",len(vectorizer_binary.get_feature_names()))
    return vectorizer_binary, X


def count_vectorize_transform(vectorizer_binary,x_list):
    X = vectorizer_binary.transform(x_list)
    return X
def elmo_transform(x_list):
   elmo = hub.load("https://tfhub.dev/google/elmo/3")
   embeddings_list = []
   vector = np.vectorize(np.float)
   for sent in x_list:
    sent = preProcessAndTokenize(sent)
    sent = ' '.join(sent) 
    #print('\nSENTENCE Length:', len(sent))
    #print('\nSENTENCE:', sent)
    #print((tf.constant(sent))
    embeddings = elmo.signatures["default"](tf.constant([sent]))
    #embeddings_list.append(vector(tf.reshape(embeddings['word_emb'],  [-1]).numpy()))
    embeddings_list.append(np.mean(embeddings['word_emb'],1).flatten())
    #elmo(sent, signature="default", as_dict=True)["elmo"]  
    #yup =tf.keras.backend.eval(embeddings)['word_emb']
   return (embeddings_list)

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


def word2vec_transform(model, x_list):
    X = np.array([np.mean([model[word] for word in filter(lambda x: x in model,preProcessAndTokenize(sent))],axis=0) for sent in x_list])    
    return X


def pretrained_word2vec_fit_transform(x_list):
    # pre trained
    # check again

    # model = Word2Vec.load(WORD2VEC_PATH)
    model = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary = True)
    X = word2vec_transform(model, x_list)
    return model , X


def selftrained_word2vec_fit(corpus):
    tokensed_corpus = [preProcessAndTokenize(sent) for sent in corpus]
    model = Word2Vec(tokensed_corpus, vector_size = 100).wv
    return model


def selftrained_word2vec_fit_transform(x_list):
    
    ## To-Do
    # change corpus to all possible texts
    corpus = x_list
    model = selftrained_word2vec_fit(corpus)
    X = word2vec_transform(model, x_list)
    
    return model , X


data = get_data_from_mongo('trial')
data = pd.DataFrame(data)
s= data['bio'][0]
s2 = data['bio'][1]
y=[s,s2]


print(y)
#elmo_transform(data['bio'])
#print(elmo_transform(y))
#print(embeddings['word_emb'])  
#print(embeddings)
X_train = elmo_transform(y)
classifier = SVC(C=1, kernel = 'linear', gamma = 'auto', class_weight=None)
#X_train = tf.reshape(X_train, [1])
print("first instance of X_train\n", type(X_train[0]),X_train[0].shape, X_train[0].ndim, X_train[0].size)

print(X_train)

classifier.fit(X_train, ['1','0'])





