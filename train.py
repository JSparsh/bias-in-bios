from config import CLASS_GROUP, TITLE, MASKED, SEED, DATASET_NAMES, PREDICTED_DATASET
from preprocessing import embedding_fit_transform, embedding_transform
from sampling import data_selection, load_data
from model import model_training, model_prediction
from evaluation import tpr_gender_gap, average_odds_difference
import pandas as pd
from joblib import dump, load



def main(load_data_from_saved, embedding_train, model_train, predict, evaluate, class_group, sampling, embedding, model, test_size, masking):
	print({'load_data_from_saved':load_data_from_saved, 'embedding_train':embedding_train, 'model_train':model_train, 'predict':predict, 'evaluate':evaluate, 'class_group':class_group, 'sampling':sampling, 'embedding':embedding, 'model':model, 'test_size':test_size, 'masking':masking})
	print("processing train.main ...")
	data = pd.DataFrame(load_data(class_group=class_group, from_saved=load_data_from_saved))

	train_set,test_set = data_selection(data, class_group, sampling, test_size, masking)

	# training 
	if model_train:
		if embedding_train:
			X_train = embedding_fit_transform(train_set[MASKED[masking]], embedding, class_group, sampling, test_size, masking)
		else:
			print("processing preprocessing.embedding_transform ...")
			X_train = embedding_transform(train_set[MASKED[masking]], embedding, class_group, sampling, test_size, masking)

		Y_train = train_set[TITLE]
		model_training(X_train, Y_train, model, embedding, class_group, sampling, test_size, masking)

	# prediction
	if predict:
		X_test = embedding_transform(test_set[MASKED[masking]], embedding, class_group, sampling, test_size, masking)
		Y_test = test_set[TITLE]

		print("processing model.model_prediction ...")
		pred, acc = model_prediction(X_test, Y_test, model, embedding, class_group, sampling, test_size, masking)
		predicted_dataset = pd.DataFrame({MASKED[masking]:test_set[MASKED[masking]], TITLE:test_set[TITLE], 'predicted':pred})

		print("\t saving file :",PREDICTED_DATASET[model, embedding, class_group, sampling, test_size, MASKED[masking]])
		dump(predicted_dataset, PREDICTED_DATASET[model, embedding, class_group, sampling, test_size, MASKED[masking]] + '.joblib')

		print("Model accuracy:", acc)
	
	# evaluation
	# TO-DO:
	# combine the prediction in evaluation to the saved prediction

	if evaluate:
		scores,x_males,y_males,x_females,y_females = tpr_gender_gap(None,test_set, model, embedding, class_group, sampling, test_size, masking)
		scores,x_males,y_males,x_females,y_females = average_odds_difference(scores,test_set, model, embedding, class_group, sampling, test_size, masking)
		print(scores)		


'''
load_data_from_saved 	-> True if saved data to be used and False if new data to be taken 
embedding_train 		-> True to train new embedding and False to use the saved one
model_train 			-> True to train new model and False to use the saved one
predict 				-> True to perform predictions on the test set and False otherwise
evaluate 				-> True to perform bias evaluations on the test set and False otherwise
class_group 			-> choice of domain of occupations ('trial','medical') 
sampling 				-> choice of sampling ('random', 'balanced')
embedding 				-> choice of embeddings to be used ('cv': count_vectorize(self-trained), 'w2v': word2vec_embedding(pre-trained), 'self_w2v':w2v(self-trained))
model 					-> choice of models to be trained ('svm', 'rf', 'nn')
test_size 				-> proportion of data to be used for tesing
masking 				-> True for 'bio' data and False for 'raw' data
'''
if __name__ == "__main__":
	import time
	start_time = time.time()
	
	import argparse
	# Initialize parser
	parser = argparse.ArgumentParser()
	 
	# Adding optional argument
	parser.add_argument('--feature', dest='feature', action='store_true')
	parser.add_argument('--no-feature', dest='feature', action='store_false')
	parser.set_defaults(feature=True)
	
	parser.add_argument("--load_data_from_saved", dest = 'load_data_from_saved',action='store_true', help = "if saved data to be used ")
	parser.add_argument("--no-load_data_from_saved", dest = 'load_data_from_saved',action='store_false', help = "if new data to be taken")
	parser.set_defaults(load_data_from_saved=False)
	
	parser.add_argument("--embedding_train", dest='embedding_train',action='store_true',help = "to train new embedding")
	parser.add_argument("--no-embedding_train", dest='embedding_train',action='store_false', help = "to use the saved embedding")
	parser.set_defaults(embedding_train=True)
	
	parser.add_argument("--model_train", dest='model_train',action='store_true', help = "to train new model")
	parser.add_argument("--no-model_train", dest='model_train',action='store_false', help = "to use the saved model")
	parser.set_defaults(model_train=True)
	
	parser.add_argument("--predict", dest='predict',action='store_true', help = "to perform predictions on the test set")
	parser.add_argument("--no-predict", dest='predict',action='store_false', help = "otherwise")
	parser.set_defaults(predict=True)
	
	parser.add_argument("--evaluate", dest='evaluate',action='store_true', help = "to perform bias evaluations on the test set")
	parser.add_argument("--no-evaluate", dest='evaluate',action='store_false', help = "otherwise")
	parser.set_defaults(evaluate=True)
	
	parser.add_argument("--masking", dest='masking',action='store_true', help = "for 'bio' data")
	parser.add_argument("--no-masking", dest='masking',action='store_false', help = "for 'raw' data")
	
	parser.add_argument("--class_group", default='medical', required=True, help = "choice of domain of occupations ('trial','medical')")
	parser.add_argument("--sampling", required=True, help = "choice of sampling ('random', 'balanced')")
	parser.add_argument("--embedding", required=True, help = "choice of embeddings to be used ('cv': count_vectorize(self-trained), 'w2v': word2vec_embedding(pre-trained), 'self_w2v':w2v(self-trained))")
	parser.add_argument("--model", default= 'svm', required=True, help = "choice of models to be trained ('svm', 'rf', 'nn')")
	parser.add_argument("--test_size", default = 0.2, required=True, help = "proportion of data to be used for tesing")
	 
	# Read arguments from command line
	args = parser.parse_args()
	 
	if args.load_data_from_saved:
	    print("Displaying load_data_from_saved as: % s" % args.load_data_from_saved)

	main(load_data_from_saved = args.load_data_from_saved, 
		embedding_train = args.embedding_train,
		model_train = args.model_train, 
		predict = args.predict,
		evaluate = args.evaluate,
		class_group = args.class_group, 
		sampling = args.sampling, 
		embedding = args.embedding, 
		model = args.model, 
		test_size = float(args.test_size), 
		masking = args.masking)
	print("\n--- %s seconds ---" % (time.time() - start_time))
