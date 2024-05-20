import torch
from constants import DEVICE

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
class ProbabilitiesSum(MultiAgentVotingRule):

    def __init__(self):
        super().__init__()

    def __call__(self, probabilities, label):
        total_probs = torch.zeros(3).to(DEVICE)

        # Summing up all probabilities
        for p in probabilities.values():
            total_probs = torch.add(total_probs, p)

        # Picking the bigger one (we want the index --> argmax)
        predicted_class = torch.argmax(total_probs).item()

        # Call the base class method to update metrics
        self.update_metrics(predicted_class, label)

        return predicted_class
    
class Plurality(MultiAgentVotingRule):

    def __init__(self):
        super().__init__()

    def __call__(self, probabilities, label):
        counter = torch.zeros(3).to(DEVICE)

        # Updating the counter tensor
        for p in probabilities.values():
            predicted = torch.argmax(p).item()
            counter[predicted] += 1
            
        # Picking the bigger one (we want the index --> argmax)
        predicted_class = torch.argmax(counter)

        # Call the base class method to update metrics
        self.update_metrics(predicted_class, label)

        return predicted_class
    
class MaxProb(MultiAgentVotingRule):

    def __init__(self):
        super().__init__()

    def __call__(self, probabilities, label):
        curr_pred_class = -1
        curr_pred_prob = 0

        # Update predicted class and probability
        for p in probabilities.values():
            predicted = torch.argmax(p).item()
            prob = torch.max(p).item()

            # Check if current predicted class has higher probability
            if prob > curr_pred_prob:
                curr_pred_prob = prob
                curr_pred_class = predicted

        # Call the base class method to update metrics
        self.update_metrics(curr_pred_class, label)

        # The predicted class is the one that remains in the variable
        # at the end of the loop
        return curr_pred_class
    
class WeightedAverage(MultiAgentVotingRule):

    def __init__(self):
        super().__init__()

    def __call__(self, probabilities, label):
        partial_sum = 0
        total_prob = 0

        # We want to make a weighted average on predicted class
        # with respect to the prediction probability
        for p in probabilities.values():
            predicted_class = torch.argmax(p).item()
            prob = torch.max(p).item()

            # Update variables
            partial_sum += predicted_class * prob
            total_prob += prob
            
        # The predicted class is the round of the weighted average
        predicted_class = round(partial_sum / total_prob)

        # Call the base class method to update metrics
        self.update_metrics(predicted_class, label)

        return predicted_class

class Borda(MultiAgentVotingRule):

    def __init__(self):
        super().__init__()

    def __call__(self, probabilities, label):
        scores = torch.zeros(3).to(DEVICE)

        # Calculate and summing up BORDA scores
        for value in probabilities.values():
            scores += torch.add(scores, torch.argsort(value)[0])

        # The predicted class is the one that has highest BORDA score
        predicted_class = torch.argmax(scores)
        
        # Call the base class method to update metrics
        self.update_metrics(predicted_class, label)

        # Call the base class method to update metrics
        return predicted_class