from config import CLASS_GROUP, MASKED, TITLE, PLOT_NAMES
from model import model_prediction
from preprocessing import embedding_transform
import numpy as np
import matplotlib.pyplot as plt

# res.class.gender.tp|fp

def tpr_fpr(test, model, embedding, class_group, sampling, test_size, masking):
	print("processing evaluation.tpr_fpr ...")
	res = {}
	for c in CLASS_GROUP[class_group]:
		res[c] = {}
		for gender in ['M','F']:
			
			mini_set = test.loc[(test['gender']==gender) & (test[TITLE]==c)]
			X_test = embedding_transform(mini_set[MASKED[masking]], embedding, class_group, sampling, test_size, masking)
			
			pred, acc = model_prediction(X_test=X_test, Y_test=mini_set[TITLE], model=model, embedding=embedding, class_group=class_group, sampling=sampling, test_size=test_size, masking=masking)
			
			tp = np.sum(pred == mini_set[TITLE])
			fp = np.sum(pred != mini_set[TITLE])

			res[c][gender] = {
								'count':len(pred), 
								'tp':tp, 
								'tpr':float(tp)/len(pred), 
								'fp':fp, 'count':len(pred), 
								'fpr':float(fp)/len(pred)}
	
	scores = {	'count_males' :[],
				'count_females' : [],
				'tpr_males' : [],
				'tpr_females' : [],
				'tp_males' : [],
				'tp_females' : [],
				'fpr_males' : [],
				'fpr_females' : [],
				'fp_males' : [],
				'fp_females' : []

				}

	for c in res:
		
		scores['count_males'].append(res[c]['M']['count'])
		scores['count_females'].append(res[c]['F']['count'])
		
		scores['tpr_males'].append(res[c]['M']['tpr'])		
		scores['tpr_females'].append(res[c]['F']['tpr'])
		scores['tp_males'].append(res[c]['M']['tp'])		
		scores['tp_females'].append(res[c]['F']['tp'])
		
		scores['fpr_males'].append(res[c]['M']['fpr'])		
		scores['fpr_females'].append(res[c]['F']['fpr'])
		scores['fp_males'].append(res[c]['M']['fp'])		
		scores['fp_females'].append(res[c]['F']['fp'])


	return(scores)


def tpr_gender_gap(scores ,test, model, embedding, class_group, sampling, test_size, masking):
	print("processing evaluation.tpr_gender_gap ...")
	if not scores: 
		scores = tpr_fpr(test, model, embedding, class_group, sampling, test_size, masking)
	
	
	x_males = [scores['count_males'][i]/(scores['count_males'][i]+scores['count_females'][i]) for i in range(len(CLASS_GROUP[class_group]))]
	y_males = [scores['tpr_males'][i]-scores['tpr_females'][i] for i in range(len(CLASS_GROUP[class_group]))]
	x_females = [scores['count_females'][i]/(scores['count_males'][i]+scores['count_females'][i]) for i in range(len(CLASS_GROUP[class_group]))]
	y_females = [scores['tpr_females'][i]-scores['tpr_males'][i] for i in range(len(CLASS_GROUP[class_group]))]
	
	fig, (axs_males,axs_females) = plt.subplots(1, 2, figsize=(20, 8))

	axs_males.scatter(x_males, y_males)
	axs_males.set_xlabel("% Male")
	axs_males.set_ylabel("TPR Gender Gap Male")
	for i, txt in enumerate(CLASS_GROUP[class_group]):
	    axs_males.annotate(txt, (x_males[i], y_males[i]))

	axs_females.scatter(x_females, y_females)
	axs_females.set_xlabel("% Female")
	axs_females.set_ylabel("TPR Gender Gap Female")
	for i, txt in enumerate(CLASS_GROUP[class_group]):
	    axs_females.annotate(txt, (x_females[i], y_females[i]))
	
	plt.savefig(PLOT_NAMES['tgp',model,embedding,class_group,sampling,test_size,MASKED[masking]] + '.png')
	plt.show()

	return (scores,x_males,y_males,x_females,y_females)


def average_odds_difference(scores, test, model, embedding, class_group, sampling, test_size, masking):
	print("processing evaluation.average_odds_difference ...")

	if not scores: 
		scores = tpr_fpr(test, model, embedding, class_group, sampling, test_size, masking)

	x_males = [scores['count_males'][i]/(scores['count_males'][i]+scores['count_females'][i]) for i in range(len(CLASS_GROUP[class_group]))]
	y_males = [(scores['fpr_males'][i]-scores['fpr_females'][i] + scores['tpr_males'][i]-scores['tpr_females'][i])/2 for i in range(len(CLASS_GROUP[class_group]))]
	x_females = [scores['count_females'][i]/(scores['count_males'][i]+scores['count_females'][i]) for i in range(len(CLASS_GROUP[class_group]))]
	y_females = [(scores['fpr_females'][i]-scores['fpr_males'][i] + scores['tpr_females'][i]-scores['tpr_males'][i])/2 for i in range(len(CLASS_GROUP[class_group]))]

	fig, (axs_males,axs_females) = plt.subplots(1, 2, figsize=(20, 8))

	axs_males.scatter(x_males, y_males)
	axs_males.set_xlabel("% Male")
	axs_males.set_ylabel("Avg odds difference Male")
	for i, txt in enumerate(CLASS_GROUP[class_group]):
	    axs_males.annotate(txt, (x_males[i], y_males[i]))

	axs_females.scatter(x_females, y_females)
	axs_females.set_xlabel("% Female")
	axs_females.set_ylabel("Avg odds difference Male")
	for i, txt in enumerate(CLASS_GROUP[class_group]):
	    axs_females.annotate(txt, (x_females[i], y_females[i]))

	plt.savefig(PLOT_NAMES['aod',model,embedding,class_group,sampling,test_size,MASKED[masking]] + '.png')
	plt.show()

	return (scores,x_males,y_males,x_females,y_females)
