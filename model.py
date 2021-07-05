from sklearn.svm import SVC
from config import DATASET_NAMES, MASKED, SEED
from joblib import dump, load
import numpy as np

def model_training(X_train, Y_train, model, embedding, class_group, sampling, test_size, masking):
	print("processing model.model_training ...")
	if model == 'svm':
		trained_model = svm_train(X_train, Y_train, sampling)
	elif model == 'rf':
		trained_model = rf_train(X_train, Y_train)
	elif model == 'nn':
		trained_model = nn_train(X_train, Y_train)

	print("\t saving file :",DATASET_NAMES['model',model,embedding,class_group,sampling,test_size,MASKED[masking]])
	dump(trained_model, DATASET_NAMES['model',model,embedding,class_group,sampling,test_size,MASKED[masking]]+'.joblib')


def model_prediction(X_test, Y_test, model, embedding, class_group, sampling, test_size, masking):
	# print("processing model.model_prediction ...")
	
	trained_model = load(DATASET_NAMES['model',model,embedding,class_group,sampling,test_size,MASKED[masking]]+'.joblib')
	
	if model == 'svm':
		pred, acc = svm_predict(X_test, Y_test, trained_model)
	elif model == 'rf':
		pred, acc = rf_predict(X_test, Y_test, trained_model)
	elif model == 'nn':
		pred, acc = nn_predict(X_test, Y_test, trained_model)
	return pred, acc


def load_model(model, embedding, class_group, sampling, test_size, masking):
	return(load(DATASET_NAMES['model',model,embedding,class_group,sampling,test_size,MASKED[masking]]+'.joblib'))



def svm_train(X_train, Y_train, sampling):
	
	if sampling == 'balanced':
		class_weight = 'balanced'
	elif sampling == 'random':
		class_weight = None

	classifier = SVC(C=1, kernel = 'linear', gamma = 'auto', class_weight=class_weight,random_state=SEED)
	# print("shape of X_train",X_train.shape)
	# print("first instance of X_train", type(X_train[0]),X_train[0])
	classifier.fit(X_train, Y_train)
	return classifier


def svm_predict(X_test, Y_test, classifier):
	prediction = classifier.predict(X_test)
	acc = np.mean(prediction == Y_test)
	# print('\nPrediction accuracy on test set using Doc2Vec :',acc)
	return prediction, acc



def rf_train(X_train, Y_train):
	'''
	check again everything !!!!!!
	'''
	factor = pd.factorize(df['raw_title'])
	df.raw_title = factor[0]
	definitions = factor[1]
	print(df.raw_title.head())
	print(definitions)

	from sklearn.ensemble import RandomForestClassifier
	from sklearn.model_selection import GridSearchCV
	param_grid_Random_forest_Tree = {
	                                "n_estimators": [10,20,30],
	                                "max_features": ["auto", "log2"],
	                                "min_samples_split": [2,4,8],
	                                "bootstrap": [True, False]
	                                                     }
	RandomForestClassifier = RandomForestClassifier()
	grid = GridSearchCV(RandomForestClassifier, param_grid_Random_forest_Tree, verbose=3, cv=5)

	grid.fit(binaryDocumentMatrix[0:4000], df['raw_title'][0:4000])
	from sklearn.ensemble import RandomForestClassifier

	n_estimators = grid.best_params_['n_estimators']
	max_features = grid.best_params_['max_features']
	min_samples_split = grid.best_params_['min_samples_split']
	bootstrap = grid.best_params_['bootstrap']

	  # creating a new model with the best parameters

	decisionTreeClassifier = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, n_jobs = -1,min_samples_split=min_samples_split, bootstrap=bootstrap)
	 # training
	decisionTreeClassifier.fit(binaryDocumentMatrix[5000:10000], df['raw_title'][5000:10000])
	y_pred_test = decisionTreeClassifier.predict(binaryDocumentMatrix[11000:16000])
	reversefactor = dict(zip(range(107),definitions))
	y_pred_test = np.vectorize(reversefactor.get)(y_pred_test)
	y_pred = np.vectorize(reversefactor.get)(df['raw_title'][11000:16000])
	print('\nPrediction accuracy :',np.mean(y_pred == y_pred_test))

def rf_predict(X_test, Y_test, model):
	## TO-DO:
	prediction = None
	acc = None
	return prediction, acc

def nn_train(X_train, Y_train):

	'''
	check again everything !!!!!!
	'''
	
	from keras import Sequential
	# The maximum number of words to be used. (most frequent)
	MAX_NB_WORDS = 50000
	# Max number of words in each bio.
	MAX_SEQUENCE_LENGTH = 250
	# This is fixed.
	EMBEDDING_DIM = 100
	# number of output classes 1 or 28 (or 29 with none)
	N_CLASSES = 28

	model = Sequential()
	model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
	model.add(SpatialDropout1D(0.2))
	model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(N_CLASSES, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	epochs = 5
	batch_size = 64

	history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

def nn_predict(X_test, Y_test, model):
	## TO-DO:
	prediction = None
	acc = None
	return prediction, acc
