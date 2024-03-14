# kaggle-Natural-Language-Processing-with-Disaster-Tweets
## Task Description
In the disaster tweets classification task, I will build some machine learning models to train and predict which tweets are about real disaster (label = 1) and which one’s aren’t (label = 0).

## Implementation
This task is implemented at [kaggle](https://www.kaggle.com/competitions/nlp-getting-started/overview) with CPU and GPU T4 x2 in Python.

## Code Description
[1-data-preprocessing-disaster-tweets.ipynb](https://github.com/ShuyeeKan/kaggle-Natural-Language-Processing-with-Disaster-Tweets/blob/main/1-data-preprocessing-disaster-tweets.ipynb): This code contains several parts of data preprocessing for training set and test set. 

[2-ml-disaster-tweets.ipynb](https://github.com/ShuyeeKan/kaggle-Natural-Language-Processing-with-Disaster-Tweets/blob/main/2-ml-disaster-tweets.ipynb): This code is to construct and train three machine learning models to classify the disaster tweets, such as XGBoost, SVM, Ransom Forest.

[4-zero-shot-classification-disaster-tweets.ipynb](https://github.com/ShuyeeKan/kaggle-Natural-Language-Processing-with-Disaster-Tweets/blob/main/4-zero-shot-classification-disaster-tweets.ipynb): In this notebook, I explore the zero-shot classification using the Hugging Face library.

## Data Description
[disaster tweets 3 tokenizers data (Testset)](https://github.com/ShuyeeKan/kaggle-Natural-Language-Processing-with-Disaster-Tweets/tree/main/disaster%20tweets%203%20tokenizers%20data%20(Testset)): Contain the preprocessed results with [1-data-preprocessing-disaster-tweets.ipynb](https://github.com/ShuyeeKan/kaggle-Natural-Language-Processing-with-Disaster-Tweets/blob/main/1-data-preprocessing-disaster-tweets.ipynb) on Test set.
  * test_e.csv: preprocessed test set with TreebankWordTokenizer
  * test_u.csv: preprocessed test set with WordPunctTokenizer
  * test_s.csv: preprocessed test set with WhitespaceTokenizer

[Prediction](https://github.com/ShuyeeKan/kaggle-Natural-Language-Processing-with-Disaster-Tweets/tree/main/Prediction): Contain the predicted results on test set and sumbit at kaggle.
  * test_prediction.csv: The best result predicted by BERT model.
  * SVM_penn_tokens_prediction.csv: The result predicted by SVM model (data tokenized by TreebankWordTokenizer).
  * zero_shot_submission.csv: The result predicted by pre-trained model in zero-shot way.

## Reference
* Tutorial document
* tf + fine-tuned model: Disaster NLP: [Keras BERT using TFHub](https://www.kaggle.com/code/xhlulu/disaster-nlp-keras-bert-using-tfhub)
* tf + fine-tuned model, data preprocessing, like removing emoji, puncuations: [NLP - EDA, Bag of Words, TF IDF, GloVe, BERT](https://www.kaggle.com/code/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert#Acknowledgements)
* pt + bert + self-defined architecture, cross-verification, can refer to the structure of the code: [BERT Baseline](https://www.kaggle.com/code/bibek777/bert-baseline)
* tf + cross-verification + ClassificationReport, EDA, text cleaning：[NLP with Disaster Tweets - EDA, Cleaning and BERT](https://www.kaggle.com/code/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert/notebook#5.-Mislabeled-Samples)
* Zero-shot + pre-trained model: [Zero Shot Classification with huggingface](https://www.kaggle.com/code/lonnieqin/zero-shot-classification-with-huggingface)
