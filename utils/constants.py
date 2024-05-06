import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 32
TEST_SAMPLES = 1000
MESSAGES = ["hate speech", "offensive language", "message approved"]