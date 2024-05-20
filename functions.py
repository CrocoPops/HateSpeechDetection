import torch
import pandas
import os
import re
import emoji
import datasets
import numpy
import nltk
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from constants import DEVICE, RANDOM_STATE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Downaload the set of words
nltk.download('words', quiet=True)
words = set(nltk.corpus.words.words())

# Function to get the pretrained models and the corresponding pretrained tokenizer
# starting from the HuggingFace hub model ID 
def get_model(hf_model):
    # Get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model)

    # We have to understand which pad_token to use
    if hf_model == 'openai-community/gpt2':
        tokenizer.pad_token = tokenizer.eos_token
    elif hf_model == 'google-bert/bert-base-uncased':
        tokenizer.pad_token = '[PAD]'

    # Setup model
    model = AutoModelForSequenceClassification.from_pretrained(hf_model, num_labels=3).to(DEVICE)
    model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer

# Function to split a pandas dataframe into train and validation HuggingFace datasets
def get_hf_datasets(df, test_size, random_state = RANDOM_STATE):
    df_train, df_val = train_test_split(df, test_size=test_size, random_state=random_state)

    train_ds = datasets.Dataset.from_pandas(df_train, preserve_index=False)
    valt_ds = datasets.Dataset.from_pandas(df_val, preserve_index=False)

    return train_ds, valt_ds

# Function to apply the tokenizer to each element of a dataframe (pandas)
def get_tokenized_datasets(df, tokenizer, test_size):

    # We want to transform the pandas dataframe in two datasets from HF
    train_ds, test_ds = get_hf_datasets(df, test_size)

    def tokenization(x):
        return tokenizer(x["text"], truncation  = True, max_length = 128)
    
    # Apply the tokenization function
    train_encoded = train_ds.map(tokenization, batched=True)
    test_encoded = test_ds.map(tokenization, batched=True)

    return train_encoded, test_encoded

# Function that cleans a string
def cleaner(text):
    text = str(text).lower() # Set all the words in lower case
    text = re.sub(r'\n', ' ', text) # Remove newline characters
    text = re.sub("@[A-Za-z0-9]+","",text) # Remove usernames
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text) # Remove the URLs
    text = " ".join(text.split()) # Remove extra white spaces
    text = ''.join(c for c in text if c not in emoji.EMOJI_DATA) # Remove emojis
    text = text.replace("#", "").replace("_", " ")
    text = text.replace("http", "")
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespaces
    
    return text

# Function that gets the cleaned dataset. It can be used with
#   - split == 'train' to get the cleaned train dataset
#   - split == 'test' to get the cleaned test dataset
# If the datasets were previously created, the function return them without creating them again
def get_dataset(filename, test_size = 0.2, random_state = RANDOM_STATE):

    # Check if are already existent
    if os.path.isfile(os.path.join('cleaned', f"cleaned_{filename}")):
        print(f'Dataset "cleaned_{filename}" already cleaned found. Using it.')

        # Read dataset from file
        df = pandas.read_csv(os.path.join(os.getcwd(), 'cleaned_datasets', f'cleaned_{filename}'))

        # Drop na --> in some dataset it can happen that we have empty texts
        df.dropna(inplace = True)
        df = df.drop(df[df.text == ''].index)

        return df
    else: 
        os.makedirs('cleaned_datasets', exist_ok=True)

        print(f'No cleaned dataset names "cleaned_{filename}" found. Creating it.')

        # Read the original dataset and clean all the texts
        df = pandas.read_csv(os.path.join(os.getcwd(), 'original_datasets', filename))
        df['text'] = df['text'].map(lambda x: cleaner(x))

        # Keep only relevant columns
        df = df[["text","labels"]]

        # Drop na --> in some dataset it can happen that we have empty texts
        df.dropna(inplace = True)
        df = df.drop(df[df.text == ''].index)
        
        df.to_csv(os.path.join(os.getcwd(), 'cleaned_datasets', f'cleaned_{filename}'), index = False)

        # Return only the wanted one
        return df

# Function that calculate the accuracy of a model on a given pandas dataframe
def get_model_accuracy(model, tokenizer, dataframe):

    # Internal function that will be used to make model prediction
    def predict(x):
        inputs    = tokenizer(x, return_tensors="pt", max_length=128, truncation=True).to(DEVICE)
        probs = torch.softmax(model(**inputs).logits, 1)
        predicted = torch.argmax(probs).item()
        return predicted

    # We add a column and for each row we calculate the model prediction
    dataframe['predicted'] = dataframe['text'].progress_map(predict)

    # We add another column where we have 1 if the prediction is the same as the label,
    # 0 otherwise
    dataframe['correct'] = numpy.where(dataframe['predicted'] == dataframe['labels'], 1, 0)

    # Confusion matrix
    cm = confusion_matrix(dataframe['labels'].tolist(), dataframe['predicted'].tolist())

    # Summing all the 1s in the correct column, we get the total number of correct prediction
    return dataframe['correct'].sum() / len(dataframe), cm

# Function to make a chart plot
def plot_chart(x_labels, y_values, title, y_label = 'Accuracy', bar_thickness = 0.2, y_ticks = None):
    X_axis = numpy.arange(len(x_labels))
    plt.rc('axes', axisbelow=True)
    plt.bar(X_axis, y_values, bar_thickness) 
    plt.xticks(X_axis, x_labels)
    if y_ticks != None:
        plt.yticks([y_ticks * i for i in range(10)])
    plt.grid(True, 'major', 'y', linewidth=0.5)
    plt.ylabel(y_label) 
    plt.title(title)  
    plt.show()