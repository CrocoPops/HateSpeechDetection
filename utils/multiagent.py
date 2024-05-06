import torch
from utils.constants import DEVICE

# accuracies is a dictionary, possible values:
# accuracies['gpt2'], accuracies['lr'], accuracies['bert']

# predictions is a dictionary, possible values:
# predictions['gpt2'], predictions['lr'], predictions['bert']
# each element returned by the dict is like (predicted_class, predicted_probabilities)
# predicted_probabilities is also a tuple (prob_class0, prob_class1, prob_class2)

models = ['bert', 'gpt2', 'lr', 'xlnet']

# BASE CLASS
class MultiAgentVotingRule:

    def __init__(self):
        self.predicted_classes = []
        self.correct_counter = 0

    def update_metrics(self, predicted_class, real_label):
        self.predicted_classes.append(predicted_class)

        if predicted_class == real_label:
            self.correct_counter += 1

# CHILD CLASSES (Each one is a different voting rule)
class WeightedAverage(MultiAgentVotingRule):
    
    def __init__(self):
        super().__init__()

    def __call__(self, accuracies, predictions, label):
        partial_sum = 0

        for model, acc in accuracies.items():
            partial_sum += predictions[model]["predicted_class"] * acc

        predicted_class = round(partial_sum / sum(accuracies.values()))
        self.update_metrics(predicted_class, label)
        return predicted_class

class ProbabilitiesSum(MultiAgentVotingRule):

    def __init__(self):
        super().__init__()

    def __call__(self, accuracies, predictions, label):
        new_probs = []
        for i in range(3): # 3 classes
            new_probs.append(sum(value['prediction_probabilities'][i] for key, value in predictions.items()))

        predicted_class = new_probs.index(max(new_probs))
        self.update_metrics(predicted_class, label)
        return predicted_class

class Plurality(MultiAgentVotingRule):

    def __init__(self):
        super().__init__()

    def __call__(self, accuracies, predictions, label):
        counter = [0, 0, 0]
        for value in predictions.values():
            counter[value['predicted_class']] += 1

        predicted_class = counter.index(max(counter))
        self.update_metrics(predicted_class, label)
        return predicted_class
