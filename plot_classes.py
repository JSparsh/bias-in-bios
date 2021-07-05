import pymongo
client = pymongo.MongoClient("mongodb+srv://root:Deployment123@clusterbiobias.4mc8e.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
collection = client['biodb']['allbio']
vocab = [t for t in collection.distinct('title') if not '_' in t]

import gensim
import pandas as pd
from sklearn.cluster import KMeans

modelPath="word_embeddings/GoogleNews-vectors-negative300.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(modelPath, binary = True)

# vocab = list(model.wv.key_to_index)
# vocab = list(collection.distinct("title"))

X = model[vocab] 
kmeans = KMeans(n_clusters=7, random_state=0).fit(X)
d = dict(zip(vocab,kmeans.labels_))
for i in sorted(d,key=d.get):
	print(i,d[i])


'''
journalist 0
photographer 0
chiropractor 1
dentist 1
dietitian 1
nurse 1
physician 1
psychologist 1
surgeon 1
architect 2
composer 2
filmmaker 2
painter 2
poet 2
accountant 3
attorney 3
paralegal 3
pastor 4
professor 4
teacher 4
dj 5
model 5
comedian 6
rapper 6

'''
