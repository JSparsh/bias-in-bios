from sklearn.model_selection import train_test_split
import pymongo
from config import MONGO_HOST, MONGO_DB, MONGO_COLLECTION, CLASS_GROUP, DATASET_NAMES, SEED
from joblib import dump, load


def get_data_from_mongo(class_group):
	print('processing sampling.get_data_from_mongo ...')

	client = pymongo.MongoClient(MONGO_HOST)
	collection = client[MONGO_DB][MONGO_COLLECTION]

	data =  list(collection.find({'$or':[{'title':title} for title in CLASS_GROUP[class_group]]}))

	print("\t saving file : ",DATASET_NAMES['datasets',class_group])
	dump(data, DATASET_NAMES['datasets',class_group]+'.joblib')
	return(data)


def load_data(class_group, from_saved=True):
	print('processing sampling.load_data ...')

	if not from_saved:
		data = get_data_from_mongo(class_group)
	else:
		data = load(DATASET_NAMES['datasets',class_group]+'.joblib')
	return data

'''
sampling -> random, balanced(upsample, downsample, weighted)
'''
def data_selection(data, class_group, sampling, test_size, masking=True):
	print('processing sampling.data_selection ...')
		
	if sampling == 'random' :
		train,test = train_test_split(data, test_size = test_size, random_state = SEED)
	# same for now
	elif sampling == 'balanced':
		
		train,test = train_test_split(data, test_size = test_size, random_state = SEED)
	
	print("\t saving file : ",DATASET_NAMES['datasets',class_group,sampling,test_size])
	dump([train,test], DATASET_NAMES['datasets',class_group,sampling,test_size]+'.joblib')
	
		
	return train,test

