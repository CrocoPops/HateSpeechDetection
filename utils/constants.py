import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 32
TEST_SAMPLES = 1000
MESSAGES = ["Hate speech", "Offensive language", "Message approved"]
RANDOM_STATE = 37
TEST_SIZE = 0.2