# Health-Project

<b>TLDR:</b> We attempt to debias clinical summaries of race, and use these summaries as inputs for treatment prediction. Three methods are used: generative adversarial networks, inverse propensity weighting, and removing race as a predictor. Overall, the models achieve fair classification (i.e. demographic parity conditioned on severity score) but have low predictive power; it should be noted that baselines perform similarly poorly, and hence must be improved.

Check out the writeup <a href="https://drive.google.com/file/d/1zgGBQSmc1mVJHEZQMTeuETzEEOHrwuQW/view?usp=sharing">here</a>!

### Exploratory Data Analysis
* `eda.ipynb`: EDA to predict race from clinical notes using logistic regression & random forest, and to predict APSIII score from clinical notes using linear regression and random forest
* `eda_bert_train.py`: Script to train BERT classifier/regressor to predict race/APSIII score
* `eda_bert_eval.py`: Script to evaluate trained BERT classifier/regressor, calculates balanced accuracy/MSE metrics

### Predicting using race

### GAN de-biasing
* `gan_pretrain.py`: Pretrains a t5-small model as an autoencoder (i.e. trains it to reproduce the input clinical note)
* `gan_train.py`: Trains a GAN by using the t5-small encoder module as the <b>generator</b>, and an MLP as the <b>discriminator</b>
* `gan_eda`: Notebook to examine produced embeddings, main finding is that the vectors for black and white patients are statistically similar to one another (using cosine similarity); however, there appears to be no difference by APSIII either (which is unintuitive)
* `gan_evaluate.ipynb`: Notebook to train a classifier to predict treatment outcome

### Reweighting using Inverse Propensity Weights
* `IPW.ipynb`: Jupyter notebook containing the entire process of processing the data, training a BERT classifier that incorporates different weights depending on the patient's propensity, and evaluates its performance.

### Removing race as a predictor 
* `BERT_no_race_distinction.ipynb`: Jupyter notebook containing the entire process of processing the data, training a BERT classifier that treats all patients as pertaining to the same race, and evaluates its performance.
