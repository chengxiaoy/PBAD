from network import *
from car_dataset import *
from config import Config
from evaluate import get_map
import copy
from predict import predict
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, MultiStepLR


# Gets the GPU if there is one, otherwise the cpu


def criterion(prediction, mask, regr, size_average=True):
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
    #     mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
    mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum()

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
    MAP = get_map(model)
    print('Dev loss: {:.4f}, map {}'.format(loss, MAP))

    return loss, MAP


def training(model, optimizer, scheduler, n_epoch):
    min_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(n_epoch):
        train_model(model, epoch, scheduler, optimizer)
        valid_loss, MAP = evaluate_model(model)
        scheduler.step(MAP)
        if valid_loss < min_loss:
            min_loss = valid_loss
            torch.save(model.state_dict(), Config.model_path)
            best_model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    model = get_model(Config.model_name)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH)
    predict(model)

    Config.expriment_id = 2
    Config.model_name = "basic_unet"
    Config.MODEL_SCALE = 1

    model = get_model(Config.model_name)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH)
    predict(model)
