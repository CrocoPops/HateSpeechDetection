import torch
import matplotlib.pyplot as plt
import datasets

from sklearn.model_selection import train_test_split, GridSearchCV
from transformers import GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification, BertTokenizer, BertForSequenceClassification, XLNetTokenizer, XLNetForSequenceClassification
from torch.utils.data import TensorDataset
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from ml_things import plot_dict, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump, load

# Custom functions
import preprocessing
import models_handler

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# accuracies is a dictionary, possible values:
# accuracies['gpt2'], accuracies['lr'], accuracies['bert']

# predictions is a dictionary, possible values:
# predictions['gpt2'], predictions['lr'], predictions['bert']
# each element returned by the dict is like (predicted_class, predicted_probabilities)
# predicted_probabilities is also a tuple (prob_class0, prob_class1, prob_class2)

models = ['bert', 'gpt2', 'lr']

def weigthed_average(accuracies, predictions):
    partial_sum = 0

    for model, acc in accuracies.items():
        partial_sum += predictions[model]["predicted_class"] * acc

    return round(partial_sum / sum(accuracies.values()))

def probabilites_sum(accuracies, predictions):
    new_probs = []
    for i in range(3): # 3 classes
        new_probs.append(sum(value['prediction_probabilities'][i] for key, value in predictions.items()))
    return new_probs.index(max(new_probs))

def plurality(accuracies, predictions):
    counter = [0, 0, 0]
    for value in predictions.values():
        counter[value['predicted_class']] += 1
    return counter.index(max(counter))