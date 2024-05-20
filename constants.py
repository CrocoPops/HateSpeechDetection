import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SAMPLES_PER_DATASET = 1000
RANDOM_STATE = 42 # For reproducibility
MESSAGES = ['Hate', 'Offensive', 'None']
CLASS2MESSAGE = {0 : 'Hate', 1 : 'Offensive', 2 : 'None'}
MESSAGE2CLASS = {'Hate' : 0, 'Offensive' : 1, 'None' : 2}