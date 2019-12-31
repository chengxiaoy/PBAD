import torch


class Config():
    N_CLASS = 8
    IMG_HEIGHT = 320
    IMG_WIDTH = 1024
    MODEL_SCALE = 8
    BATCH_SIZE = 4
    DATA_PATH = './pku-autonomous-driving/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "basic"
    N_EPOCH = 10
    expriment_id = 1
    model_path = str(expriment_id) + '_model.pth'
