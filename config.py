import torch


class Config():
    N_CLASS = 8
    IMG_HEIGHT = 320
    IMG_WIDTH = 1024
    MODEL_SCALE = 8
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    PARALLEL = False
    device_ids = [0, 1]
    BATCH_SIZE = 4 * len(device_ids) if PARALLEL else 4
    DATA_PATH = './pku-autonomous-driving/'
    model_name = "basic"
    N_EPOCH = 20
    expriment_id = 1
    model_path = str(expriment_id) + '_model.pth'
    MASK_WEIGHT = 100
    FOCAL_ALPHA = 0.5
