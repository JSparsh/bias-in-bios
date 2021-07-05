Make sure these directories are present

`mkdir datasets models plots_and_graphs predicted_datasets word_embeddings`

to add pretrained word embedding models, write in terminal:

`cd word_embeddings` 

`wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"`

`gzip -d GoogleNews-vectors-negative300.bin.gz`

to get debiased version of pretrained word2Vec embedding, use the link: <br /> https://drive.google.com/file/d/1_PvT4ZvtZjhq4HPywA8-u06epht9ccOw/view?usp=sharing

install requirements.txt


config.py - contains constants

evaluation.py - contains evaluation metrics

model.py - contains model descriptions and training and prediction modules

preprocessing.py - contains embeddings and preprocessing tasks

sampling.py - contains data extraction and sampling tasks

train.py - contains the runnable flow of the project 



`python train.py --no-load_data_from_saved --embedding_train --model_train --predict --evaluate --masking --class_group medical --sampling random --embedding cv --model svm --test_size 0.2`