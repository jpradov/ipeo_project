import torch

PATH_TO_DATA = r"C:\data_repository\IPEO\project\dataset"  # path to your folder containing the images & labels folder as well as the train test split
device = "cuda" if torch.cuda.is_available() else "cpu"
