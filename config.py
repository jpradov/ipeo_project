import torch

# path to your folder containing the images & labels folder as well as the train test split
PATH_TO_DATA = r"C:\data_repository\IPEO\project\dataset"
CHECKPOINT_PATH = r"./checkpoint.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
