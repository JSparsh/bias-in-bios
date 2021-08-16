# BIAS IN BIOS

This project is created by [Nishtha Jain](https://git.rwth-aachen.de/nishthajain1611) and [Sparsh Jauhari](https://git.rwth-aachen.de/sparsh.jauhari) under the guidance of our mentor [Markus Strohmaier](http://markusstrohmaier.info/) as a lab project for "Lab Course Data, Society and Algorithms",
 Chair for Computational Social Sciences and Humanities, RWTH Aachen University.

In this project, we aim to find potential bias in the data of biographies of individuals ([BIOS.pkl](https://github.com/microsoft/biosbias)) and also in the machine learning models trained on them to predict the occupation. 




## Installation and prerequisites
```bash
pip install -r requirements.txt
```

Add these directories
```bash
mkdir datasets models predicted_datasets word_embeddings
```
Download the following embeddings for pre-trained Word2Vec embeddings in the word_embeddings directory.
```bash
cd word_embeddings 
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
gzip -d GoogleNews-vectors-negative300.bin.gz
```
To get debiased version of pretrained Word2Vec embedding, use this [link](https://drive.google.com/file/d/1_PvT4ZvtZjhq4HPywA8-u06epht9ccOw/view?usp=sharing), or download them by using [gdown](https://pypi.org/project/gdown/3.3.1/). Download them in the word_embeddings directory.

```bash
gdown https://drive.google.com/uc?id=0B5vZVlu2WoS5ZTBSekpUX0RSNDg
```
Download the debiased version of our self trained word2Vec embeddings in the word_embeddings directory.

```bash
gdown https://drive.google.com/uc?id=1Lj_RteEF_wAEsENZBLQYGBLMY99VfI5j
```


## File structure

>`config.py` - contains constants and paths
>
>`model.py` - contains model descriptions and training and prediction modules
>
>`preprocessing.py` - contains embeddings and preprocessing tasks
>
>`sampling.py` - contains data extraction and sampling tasks
>
>`train.py` - contains the runnable flow of the project 
>
>`predict.py` - contains functions to predict the occupation of given bios
>
>`bios_bias.ipynb` - contains results, evaluation metrics and plots



## Usage
The runnable `train.py` can be used to train the models and predict then on test set while
`predict.py` can be used for single predictions. All the results, plots and evaluations can be seen in the `bios_bias.ipynb`. `debiase.py` takes care of the debiasing.

#### Training and Predictions
```bash
python train.py --no-load_data_from_saved --embedding_train --model_train --predict --masking --class_group medical --sampling balanced --embedding cv --model svm --test_size 0.2
```

>`--load_data_from_saved` - if saved data to be used 
>
>`--no-load_data_from_saved` - if new data to be taken
>
>`--embedding_train` - to train new embedding
>
>`--no-embedding_train` - to use the saved embedding
>
>`--model_train` - to train new model
>
>`--no-model_train` - to use the saved model
>
>`--predict` - to perform predictions on the test set
>
>`--no-predict` - to not perform predictions on the test set
>
>`--masking` - for 'bio' data
>
>`--no-masking` - for 'raw' data
>
>`--class_group` - choice of domain of occupations ('trial','medical')
>
>`--sampling` - choice of sampling / class weights ('random', 'balanced')
>
>`--embedding` - choice of embeddings to be used ('cv':count vectorize, 'w2v':word2vec, 'self_w2v': self-trained word2vec, 'elmo':elmo, 'd_w2v:debiased word2vec, 'd_self_w2v': debiased self-trained word2vec)
>
>`--model` - choice of models to be trained ('svm')
>
>`--test_size` - proportion of data to be used for testing

#### Single predictions

To predict your own sentence on the trained models
```bash
python predict
```
>`--masking` - for 'bio' data
>
>`--no-masking` - for 'raw' data
>
>`--sampling` - choice of sampling / class weights ('random', 'balanced')
>
>`--embedding` - choice of embeddings to be used ('cv':count vectorize, 'w2v':word2vec, 'self_w2v': self-trained word2vec, 'elmo':elmo, 'd_w2v:debiased word2vec, 'd_self_w2v': debiased self-trained word2vec)
>
>`--pred_all_models` - yes for all or no for the specific one

#### Debiasing of embeddings
The `debias.py` is used to debias the word2Vec embeddings trained on our dataset of medical domain. The code in the file is a refactored version of the code that can be found at:<br /> https://github.com/tolga-b/debiaswe/tree/10277b23e187ee4bd2b6872b507163ef4198686b/debiaswe  <br />

The code uses
>`definitional_pairs.json` - The ten pairs of words used to define the gender direction. <br />
>`gender_specific_full.json` . A list of 1441 gender-specific words <br />

The source of the files is :<br /> https://github.com/tolga-b/debiaswe/tree/10277b23e187ee4bd2b6872b507163ef4198686b/data <br />


