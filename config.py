MONGO_HOST = "mongodb+srv://root:Deployment123@clusterbiobias.4mc8e.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
MONGO_DB = 'biodb'
MONGO_COLLECTION = 'allbio'

TITLE = 'title'

CLASS_GROUP = {'medical' : ['physician',
							'nurse',
							'psychologist',
							'dentist',
							'surgeon',
							'dietitian',	
							'chiropractor'
						],
				'trial' : ['yoga_teacher',
							'personal_trainer']
		}

SEED = 414325

WORD2VEC_PATH = "word_embeddings/GoogleNews-vectors-negative300.bin"

MASKED = {		True:'bio',
				False:'raw'
		}

DATASET_NAMES = { 
# Naming convention : [dset|embd|modl]_[sv|rf|nn]_[cv|wv|tv]_[tri|med]_[ran|bal]_[test_spit]_[b|r]

# trial domain
	('datasets','trial') 								: 'datasets/dset___tri___',
	('datasets','trial','random',0.2) 					: 'datasets/dset___tri_ran_0.2_',
	('datasets','trial','balanced',0.2) 				: 'datasets/dset___tri_bal_0.2_',

	('embedding','cv','trial','random',0.2,'bio') 		: 'word_embeddings/embd__cv_tri_ran_0.2_b',
	('embedding','cv','trial','random',0.2,'raw') 		: 'word_embeddings/embd__cv_tri_ran_0.2_r',
	('embedding','w2v','trial','random',0.2,'bio') 		: 'word_embeddings/embd__wv_tri_ran_0.2_b',
	('embedding','w2v','trial','random',0.2,'raw') 		: 'word_embeddings/embd__wv_tri_ran_0.2_r',
	('embedding','cv','trial','balanced',0.2,'bio') 	: 'word_embeddings/embd__cv_tri_bal_0.2_b',
	('embedding','w2v','trial','balanced',0.2,'bio') 	: 'word_embeddings/embd__wv_tri_bal_0.2_b',
	('embedding','self_w2v','trial','balanced',0.2,'bio') 	: 'word_embeddings/embd__tv_tri_bal_0.2_b',

	('model','svm','cv','trial','random',0.2,'bio') 	: 'models/modl_sv_cv_tri_ran_0.2_b',
	('model','svm','cv','trial','random',0.2,'raw') 	: 'models/modl_sv_cv_tri_ran_0.2_r',
	('model','svm','w2v','trial','random',0.2,'bio') 	: 'models/modl_sv_wv_tri_ran_0.2_b',
	('model','svm','w2v','trial','random',0.2,'raw') 	: 'models/modl_sv_wv_tri_ran_0.2_r',
	('model','svm','cv','trial','balanced',0.2,'bio') 	: 'models/modl_sv_cv_tri_bal_0.2_b',
	('model','svm','w2v','trial','balanced',0.2,'bio') 	: 'models/modl_sv_wv_tri_bal_0.2_b',
	('model','svm','self_w2v','trial','balanced',0.2,'bio') 	: 'models/modl_sv_tv_tri_bal_0.2_b',



# medical domain
	('datasets','medical') 								: 'datasets/dset___med___',
	
	('datasets','medical','random',0.2) 				: 'datasets/dset___med_ran_0.2_',
	('datasets','medical','balanced',0.2) 				: 'datasets/dset___med_bal_0.2_',

	('embedding','cv','medical','random',0.2,'bio') 	: 'word_embeddings/embd__cv_med_ran_0.2_b',
	('embedding','cv','medical','random',0.2,'raw') 	: 'word_embeddings/embd__cv_med_ran_0.2_r',
	('embedding','w2v','medical','random',0.2,'bio') 	: 'word_embeddings/embd__wv_med_ran_0.2_b',
	('embedding','w2v','medical','random',0.2,'raw') 	: 'word_embeddings/embd__wv_med_ran_0.2_r',
	('embedding','self_w2v','medical','random',0.2,'bio') 	: 'word_embeddings/embd__tv_med_ran_0.2_b',
	('embedding','self_w2v','medical','random',0.2,'raw') 	: 'word_embeddings/embd__tv_med_ran_0.2_r',
	
	('embedding','cv','medical','balanced',0.2,'bio') 	: 'word_embeddings/embd__cv_med_bal_0.2_b',
	('embedding','cv','medical','balanced',0.2,'raw') 	: 'word_embeddings/embd__cv_med_bal_0.2_r',
	('embedding','w2v','medical','balanced',0.2,'bio') 	: 'word_embeddings/embd__wv_med_bal_0.2_b',
	('embedding','w2v','medical','balanced',0.2,'raw') 	: 'word_embeddings/embd__wv_med_bal_0.2_r',
	('embedding','self_w2v','medical','balanced',0.2,'bio') 	: 'word_embeddings/embd__tv_med_bal_0.2_b',
	('embedding','self_w2v','medical','balanced',0.2,'raw') 	: 'word_embeddings/embd__tv_med_bal_0.2_r',

	('model','svm','cv','medical','random',0.2,'bio') 	: 'models/modl_sv_cv_med_ran_0.2_b',
	('model','svm','cv','medical','random',0.2,'raw') 	: 'models/modl_sv_cv_med_ran_0.2_r',
	('model','svm','w2v','medical','random',0.2,'bio') 	: 'models/modl_sv_wv_med_ran_0.2_b',
	('model','svm','w2v','medical','random',0.2,'raw') 	: 'models/modl_sv_wv_med_ran_0.2_r',
	('model','svm','self_w2v','medical','random',0.2,'bio') 	: 'models/modl_sv_tv_med_ran_0.2_b',
	('model','svm','self_w2v','medical','random',0.2,'raw') 	: 'models/modl_sv_tv_med_ran_0.2_r',
	
	('model','svm','cv','medical','balanced',0.2,'bio') 	: 'models/modl_sv_cv_med_bal_0.2_b',
	('model','svm','cv','medical','balanced',0.2,'raw') 	: 'models/modl_sv_cv_med_bal_0.2_r',
	('model','svm','w2v','medical','balanced',0.2,'bio') 	: 'models/modl_sv_wv_med_bal_0.2_b',
	('model','svm','w2v','medical','balanced',0.2,'raw') 	: 'models/modl_sv_wv_med_bal_0.2_r',
	('model','svm','self_w2v','medical','balanced',0.2,'bio') 	: 'models/modl_sv_tv_med_bal_0.2_b',
	('model','svm','self_w2v','medical','balanced',0.2,'raw') 	: 'models/modl_sv_tv_med_bal_0.2_r'

}

PREDICTED_DATASET = {
	
# Naming convention : pred_[sv|rf|nn]_[cv|wv|tv]_[tri|med]_[ran|bal]_[test_spit]_[b|r]
	
	('svm','cv','trial','balanced',0.2,'raw') 	: 'predicted_datasets/pred_sv_cv_tri_bal_0.2_r',
	('svm','cv','trial','balanced',0.2,'bio') 	: 'predicted_datasets/pred_sv_cv_tri_bal_0.2_b',
	('svm','w2v','trial','balanced',0.2,'raw') 	: 'predicted_datasets/pred_sv_wv_tri_bal_0.2_r',
	('svm','w2v','trial','balanced',0.2,'bio') 	: 'predicted_datasets/pred_sv_wv_tri_bal_0.2_b',
	('svm','self_w2v','trial','balanced',0.2,'raw') 	: 'predicted_datasets/pred_sv_tv_tri_bal_0.2_r',
	('svm','self_w2v','trial','balanced',0.2,'bio') 	: 'predicted_datasets/pred_sv_tv_tri_bal_0.2_b',
	
	('svm','cv','trial','random',0.2,'raw') 	: 'predicted_datasets/pred_sv_cv_tri_ran_0.2_r',
	('svm','cv','trial','random',0.2,'bio') 	: 'predicted_datasets/pred_sv_cv_tri_ran_0.2_b',
	('svm','w2v','trial','random',0.2,'raw') 	: 'predicted_datasets/pred_sv_wv_tri_ran_0.2_r',
	('svm','w2v','trial','random',0.2,'bio') 	: 'predicted_datasets/pred_sv_wv_tri_ran_0.2_b',
	('svm','self_w2v','trial','random',0.2,'raw') 	: 'predicted_datasets/pred_sv_tv_tri_ran_0.2_r',
	('svm','self_w2v','trial','random',0.2,'bio') 	: 'predicted_datasets/pred_sv_tv_tri_ran_0.2_b',

	
	

	('svm','cv','medical','random',0.2,'raw') 	: 'predicted_datasets/pred_sv_cv_med_ran_0.2_r',
	('svm','cv','medical','random',0.2,'bio') 	: 'predicted_datasets/pred_sv_cv_med_ran_0.2_b',
	('svm','w2v','medical','random',0.2,'raw') 	: 'predicted_datasets/pred_sv_wv_med_ran_0.2_r',
	('svm','w2v','medical','random',0.2,'bio') 	: 'predicted_datasets/pred_sv_wv_med_ran_0.2_b',
	('svm','self_w2v','medical','random',0.2,'raw') 	: 'predicted_datasets/pred_sv_tv_med_ran_0.2_r',
	('svm','self_w2v','medical','random',0.2,'bio') 	: 'predicted_datasets/pred_sv_tv_med_ran_0.2_b',
	
	('svm','cv','medical','balanced',0.2,'raw') 	: 'predicted_datasets/pred_sv_cv_med_bal_0.2_r',
	('svm','cv','medical','balanced',0.2,'bio') 	: 'predicted_datasets/pred_sv_cv_med_bal_0.2_b',
	('svm','w2v','medical','balanced',0.2,'raw') 	: 'predicted_datasets/pred_sv_wv_med_bal_0.2_r',
	('svm','w2v','medical','balanced',0.2,'bio') 	: 'predicted_datasets/pred_sv_wv_med_bal_0.2_b',
	('svm','self_w2v','medical','balanced',0.2,'raw') 	: 'predicted_datasets/pred_sv_tv_med_bal_0.2_r',
	('svm','self_w2v','medical','balanced',0.2,'bio') 	: 'predicted_datasets/pred_sv_tv_med_bal_0.2_b'

}


PLOT_NAMES = {
	
# Naming convention : plot_[tgp|aod]_[M|F]_[sv|rf|nn]_[cv|wv|tv]_[tri|med]_[ran|bal]_[test_spit]_[b|r]

	('tgp','svm','self_w2v','trial','balanced',0.2,'bio') : 'plots_and_graphs/plot_tgp_sv_tv_tri_bal_0.2_b',
	('aod','svm','self_w2v','trial','balanced',0.2,'bio') : 'plots_and_graphs/plot_aod_sv_tv_tri_bal_0.2_b',
	
	('tgp','svm','cv','medical','random',0.2,'raw') : 'plots_and_graphs/plot_tgp_sv_cv_med_ran_0.2_r',
	('tgp','svm','cv','medical','random',0.2,'bio') : 'plots_and_graphs/plot_tgp_sv_cv_med_ran_0.2_b',
	('tgp','svm','w2v','medical','random',0.2,'raw') : 'plots_and_graphs/plot_tgp_sv_wv_med_ran_0.2_r',
	('tgp','svm','w2v','medical','random',0.2,'bio') : 'plots_and_graphs/plot_tgp_sv_wv_med_ran_0.2_b',
	('tgp','svm','self_w2v','medical','random',0.2,'raw') : 'plots_and_graphs/plot_tgp_sv_tv_med_ran_0.2_r',
	('tgp','svm','self_w2v','medical','random',0.2,'bio') : 'plots_and_graphs/plot_tgp_sv_tv_med_ran_0.2_b',

	('tgp','svm','cv','medical','balanced',0.2,'raw') : 'plots_and_graphs/plot_tgp_sv_cv_med_bal_0.2_r',
	('tgp','svm','cv','medical','balanced',0.2,'bio') : 'plots_and_graphs/plot_tgp_sv_cv_med_bal_0.2_b',
	('tgp','svm','w2v','medical','balanced',0.2,'raw') : 'plots_and_graphs/plot_tgp_sv_wv_med_bal_0.2_r',
	('tgp','svm','w2v','medical','balanced',0.2,'bio') : 'plots_and_graphs/plot_tgp_sv_wv_med_bal_0.2_b',
	('tgp','svm','self_w2v','medical','balanced',0.2,'raw') : 'plots_and_graphs/plot_tgp_sv_tv_med_bal_0.2_r',
	('tgp','svm','self_w2v','medical','balanced',0.2,'bio') : 'plots_and_graphs/plot_tgp_sv_tv_med_bal_0.2_b',
	
	
	('aod','svm','cv','medical','random',0.2,'raw') : 'plots_and_graphs/plot_aod_sv_cv_med_ran_0.2_r',
	('aod','svm','cv','medical','random',0.2,'bio') : 'plots_and_graphs/plot_aod_sv_cv_med_ran_0.2_b',
	('aod','svm','w2v','medical','random',0.2,'raw') : 'plots_and_graphs/plot_aod_sv_wv_med_ran_0.2_r',
	('aod','svm','w2v','medical','random',0.2,'bio') : 'plots_and_graphs/plot_aod_sv_wv_med_ran_0.2_b',
	('aod','svm','self_w2v','medical','random',0.2,'raw') : 'plots_and_graphs/plot_aod_sv_tv_med_ran_0.2_r',
	('aod','svm','self_w2v','medical','random',0.2,'bio') : 'plots_and_graphs/plot_aod_sv_tv_med_ran_0.2_b',

	('aod','svm','cv','medical','balanced',0.2,'raw') : 'plots_and_graphs/plot_aod_sv_cv_med_bal_0.2_r',
	('aod','svm','cv','medical','balanced',0.2,'bio') : 'plots_and_graphs/plot_aod_sv_cv_med_bal_0.2_b',
	('aod','svm','w2v','medical','balanced',0.2,'raw') : 'plots_and_graphs/plot_aod_sv_wv_med_bal_0.2_r',
	('aod','svm','w2v','medical','balanced',0.2,'bio') : 'plots_and_graphs/plot_aod_sv_wv_med_bal_0.2_b',
	('aod','svm','self_w2v','medical','balanced',0.2,'raw') : 'plots_and_graphs/plot_aod_sv_tv_med_bal_0.2_r',
	('aod','svm','self_w2v','medical','balanced',0.2,'bio') : 'plots_and_graphs/plot_aod_sv_tv_med_bal_0.2_b'

}
