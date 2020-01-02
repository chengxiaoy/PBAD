from network import *
from car_dataset import *
from config import Config
from evaluate import get_map
import copy
from predict import predict
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, MultiStepLR
from loss import FocalLoss


# Gets the GPU if there is one, otherwise the cpu


def criterion(prediction, mask, regr, size_average=True):
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])

    #     mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
    # mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    # mask_loss = -mask_loss.mean(0).sum()

    # focal loss
    mask_criterion = FocalLoss()
    mask_loss = mask_criterion(pred_mask, mask)

    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)

    # Sum
    loss = mask_loss + regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss


def train_model(model, epoch, scheduler, optimizer):
    model.train()
    epoch_loss = 0

    for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(tqdm(train_loader)):
        img_batch = img_batch.to(Config.device)
        mask_batch = mask_batch.to(Config.device)
        regr_batch = regr_batch.to(Config.device)

        optimizer.zero_grad()
        output = model(img_batch)
        loss = criterion(output, mask_batch, regr_batch)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(train_loader)
    print('Train Epoch: {} \tLoss: {}'.format(
        epoch,
        epoch_loss))

    return epoch_loss


def evaluate_model(model):
    model.eval()
    loss = 0

    with torch.no_grad():
        for img_batch, mask_batch, regr_batch in valid_loader:
            img_batch = img_batch.to(Config.device)
            mask_batch = mask_batch.to(Config.device)
            regr_batch = regr_batch.to(Config.device)
            output = model(img_batch)
            loss += criterion(output, mask_batch, regr_batch, size_average=False).item()

    loss /= len(valid_loader.dataset)
    try:
        MAP = get_map(model)
    except Exception as e:
        MAP = 0.0
    print('Dev loss: {:.4f}, map {}'.format(loss, MAP))

    return loss, MAP


def training(model, optimizer, scheduler, n_epoch):
    min_loss = float('inf')
    max_MAP = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(n_epoch):
        train_model(model, epoch, scheduler, optimizer)
        valid_loss, MAP = evaluate_model(model)
        scheduler.step(valid_loss)
        if MAP > max_MAP:
            max_MAP = MAP
            torch.save(model.state_dict(), Config.model_path)
            best_model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    Config.expriment_id = 3

    model = get_model(Config.model_name)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH)
    predict(model)

    # Config.expriment_id = 2
    # Config.model_name = "basic_unet"
    # Config.MODEL_SCALE = 1
    # Config.BATCH_SIZE = 8
    #
    # model = get_model(Config.model_name)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH)
    # predict(model)
