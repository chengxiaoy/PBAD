from network import *
from car_dataset import *
from config import Config
from evaluate import get_map
import copy
from predict import predict
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, MultiStepLR
from loss import FocalLoss
from tensorboardX import SummaryWriter
from loss import UncertaintyLoss


# Gets the GPU if there is one, otherwise the cpu

def hourglass_criterion(predictions, mask, regr, uncertain_loss, batch_idx, size_average=True):
    if Config.model_name == 'hourglass':
        sum_loss = 0.0
        for idx in range(len(predictions)):
            output = torch.cat((predictions[idx]['mp'], predictions[idx]['xyz'], predictions[idx]['roll']), dim=1)
            loss = criterion(output, mask, regr, uncertain_loss, batch_idx, size_average)

            sum_loss += loss / len(predictions)
        return sum_loss


def criterion(prediction, mask, regr, uncertain_loss, batch_idx, size_average=True):
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])

    #     mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
    # mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    # mask_loss = -mask_loss.mean(0).sum()

    # focal loss
    mask_criterion = FocalLoss(alpha=Config.FOCAL_ALPHA)
    mask_loss = mask_criterion(pred_mask, mask)

    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)

    if batch_idx % 500 == 0:
        print("mask loss{}".format(mask_loss))
        print("regr loss{}".format(regr_loss))

    # Sum

    if not Config.USE_UNCERTAIN_LOSS:
        loss = Config.MASK_WEIGHT * mask_loss + regr_loss
    else:
        loss = uncertain_loss(Config.MASK_WEIGHT * mask_loss, regr_loss)

    if not size_average:
        loss *= prediction.shape[0]
    return loss


def train_model(model, epoch, uncertain_loss, optimizer):
    model.train()
    epoch_loss = 0

    train_loader = get_data_loader()[0]

    for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(tqdm(train_loader)):

        img_batch = img_batch.to(Config.device)
        mask_batch = mask_batch.to(Config.device)
        regr_batch = regr_batch.to(Config.device)

        optimizer.zero_grad()
        output = model(img_batch)
        if Config.model_name.startswith('dla'):
            output = torch.cat((output[0]['mp'], output[0]['xyz'], output[0]['roll']), dim=1)

        if Config.model_name == 'hourglass':
            loss = hourglass_criterion(output, mask_batch, regr_batch, uncertain_loss, batch_idx)
        else:
            loss = criterion(output, mask_batch, regr_batch, uncertain_loss, batch_idx)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(train_loader)
    print('Train Epoch: {} \tLoss: {}'.format(
        epoch,
        epoch_loss))

    return epoch_loss


def evaluate_model(model, uncertain_loss):
    model.eval()
    loss = 0
    valid_loader = get_data_loader()[1]

    with torch.no_grad():
        for img_batch, mask_batch, regr_batch in valid_loader:
            img_batch = img_batch.to(Config.device)
            mask_batch = mask_batch.to(Config.device)
            regr_batch = regr_batch.to(Config.device)
            output = model(img_batch)
            if Config.model_name.startswith('dla'):
                output = torch.cat((output[0]['mp'], output[0]['xyz'], output[0]['roll']), dim=1)

            if Config.model_name == 'hourglass':
                loss += hourglass_criterion(output, mask_batch, regr_batch, uncertain_loss, batch_idx=1,
                                            size_average=False)
            else:
                loss += criterion(output, mask_batch, regr_batch, uncertain_loss, batch_idx=1, size_average=False)

    loss /= len(valid_loader.dataset)
    MAP = get_map(model)

    print('Dev loss: {:.4f}, map {}'.format(loss, MAP))

    return loss, MAP


def training(model, optimizer, scheduler, n_epoch, writer, uncertain_loss):
    min_loss = float('inf')
    max_MAP = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(n_epoch):
        train_loss = train_model(model, epoch, uncertain_loss, optimizer)
        valid_loss, MAP = evaluate_model(model, uncertain_loss)
        scheduler.step(MAP)

        writer.add_scalars('data/loss', {'train': train_loss, 'val': valid_loss}, epoch)
        writer.add_scalars('data/map', {'val': MAP}, epoch)

        if MAP > max_MAP:
            max_MAP = MAP
            torch.save(model.state_dict(), str(Config.expriment_id) + '_model.pth')
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    # Config.expriment_id = 30_2
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "unet"
    # Config.MODEL_SCALE = 4
    # # Config.IMG_WIDTH = 1536
    # # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 30
    # Config.MASK_WEIGHT = 500
    # Config.USE_UNCERTAIN_LOSS = True
    # Config.USE_MASK = True
    # model = get_model(Config.model_name)
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # params = list(uncertain_loss.parameters()) + list(model.parameters())
    # optimizer = optim.Adam(params, lr=0.001)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)

    # Config.expriment_id = 30_3
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "unet"
    # Config.MODEL_SCALE = 4
    # # Config.IMG_WIDTH = 1536
    # # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 30
    # Config.MASK_WEIGHT = 1
    # Config.USE_UNCERTAIN_LOSS = True
    # Config.USE_MASK = True
    # model = get_model(Config.model_name)
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # params = list(uncertain_loss.parameters()) + list(model.parameters())
    # optimizer = optim.Adam(params, lr=0.001)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)

    # Config.expriment_id = 30_5
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "dla34"
    # Config.MODEL_SCALE = 4
    # # Config.IMG_WIDTH = 1536
    # # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 30
    # Config.MASK_WEIGHT = 200
    # Config.USE_UNCERTAIN_LOSS = True
    # Config.USE_MASK = True
    # model = get_model(Config.model_name)
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # params = list(uncertain_loss.parameters()) + list(model.parameters())
    # optimizer = optim.AdamW(params, lr=0.001, weight_decay=0.01)
    #
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)

    # Config.expriment_id = 30_6
    # Config.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "dla34_2"
    # Config.MODEL_SCALE = 4
    # # Config.IMG_WIDTH = 1536
    # # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 30
    # Config.MASK_WEIGHT = 200
    # Config.USE_UNCERTAIN_LOSS = True
    # Config.USE_MASK = True
    # model = get_model(Config.model_name)
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # params = list(uncertain_loss.parameters()) + list(model.parameters())
    # optimizer = optim.AdamW(params, lr=0.001, weight_decay=0.01)
    #
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)

    # Config.expriment_id = 30_61
    #
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "dla34_2"
    # Config.MODEL_SCALE = 4
    # # Config.IMG_WIDTH = 1536
    # # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 10
    # Config.MASK_WEIGHT = 200
    # Config.USE_UNCERTAIN_LOSS = True
    # Config.USE_MASK = True
    # model = get_model(Config.model_name)
    # model.load_state_dict(torch.load('306_model.pth'))
    #
    #
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # params = list(uncertain_loss.parameters()) + list(model.parameters())
    # optimizer = optim.AdamW(params, lr=0.0001, weight_decay=0.01)
    #
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=2, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)

    # Config.expriment_id = 30_7
    #
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "dla102_x"
    # Config.MODEL_SCALE = 4
    # # Config.IMG_WIDTH = 1536
    # # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 30
    # Config.MASK_WEIGHT = 200
    # Config.USE_UNCERTAIN_LOSS = True
    # Config.USE_MASK = True
    # model = get_model(Config.model_name)
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # params = list(uncertain_loss.parameters()) + list(model.parameters())
    # optimizer = optim.AdamW(params, lr=0.001, weight_decay=0.01)
    #
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[12, 20], gamma=0.1)
    # model = training(model, optimizer, scheduler=scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)

    # Config.expriment_id = 10_9
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic_4"
    # Config.MODEL_SCALE = 4
    # # Config.IMG_WIDTH = 1536
    # # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 30
    # Config.MASK_WEIGHT = 200
    # Config.USE_UNCERTAIN_LOSS = True
    # Config.USE_MASK = True
    # Config.USE_GAUSSIAN = True
    # model = get_model(Config.model_name)
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # params = list(uncertain_loss.parameters()) + list(model.parameters())
    # optimizer = optim.AdamW(params, lr=0.001, weight_decay=0.01)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)
    #
    # Config.expriment_id = 15_1
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "hourglass"
    # Config.MODEL_SCALE = 4
    # Config.IMG_WIDTH = 1536
    # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 30
    # Config.MASK_WEIGHT = 500
    # Config.USE_UNCERTAIN_LOSS = False
    # Config.USE_MASK = True
    # Config.USE_GAUSSIAN = False
    # model = get_model(Config.model_name)
    # model.load_state_dict(torch.load('15_model.pth'))
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # params = list(uncertain_loss.parameters()) + list(model.parameters())
    # optimizer = optim.AdamW(params, lr=0.0001, weight_decay=0.01)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)

    # Config.expriment_id = 31
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "unet"
    # Config.MODEL_SCALE = 4
    # Config.IMG_WIDTH = 1536
    # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 30
    # Config.MASK_WEIGHT = 100
    # Config.USE_UNCERTAIN_LOSS = True
    # Config.USE_MASK = True
    # model = get_model(Config.model_name)
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # params = list(uncertain_loss.parameters()) + list(model.parameters())
    # optimizer = optim.AdamW(params, lr=0.001, weight_decay=0.01)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)

    # Config.expriment_id = 17_1
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic_4"
    # Config.MODEL_SCALE = 4
    # Config.IMG_WIDTH = 1536
    # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.8
    # Config.N_EPOCH = 10
    # Config.MASK_WEIGHT = 500
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # Config.USE_MASK = True
    # model = get_model(Config.model_name)
    # model.load_state_dict(torch.load('17_model.pth'))
    #
    # optimizer = optim.AdamW(model.parameters(), lr=0.00003, weight_decay=0.01)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)
    #
    # Config.expriment_id = 19
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic_4"
    # Config.MODEL_SCALE = 4
    # Config.IMG_WIDTH = 1536
    # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.8
    # Config.N_EPOCH = 40
    # Config.MASK_WEIGHT = 500
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # Config.USE_MASK = False
    # Config.FOUR_CHANNEL = True
    # model = get_model(Config.model_name)
    # # model.load_state_dict(torch.load('17_model.pth'))
    #
    # optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)


    # Config.expriment_id = 20
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic_4"
    # Config.MODEL_SCALE = 4
    # Config.IMG_WIDTH = 1536
    # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.8
    # Config.N_EPOCH = 40
    # Config.MASK_WEIGHT = 500
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # Config.USE_MASK = True
    # Config.FOUR_CHANNEL = False
    # Config.USE_UNCERTAIN_LOSS = True
    # model = get_model(Config.model_name)
    # # model.load_state_dict(torch.load('17_model.pth'))
    #
    # optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)


    # Config.expriment_id = 30_19
    # Config.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "dla34_2"
    # Config.MODEL_SCALE = 4
    # Config.BATCH_SIZE = 4
    # # Config.IMG_WIDTH = 1536
    # # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 30
    # Config.MASK_WEIGHT = 0.1
    # Config.USE_UNCERTAIN_LOSS = False
    # Config.USE_MASK = True
    # Config.USE_GAUSSIAN = True
    # model = get_model(Config.model_name)
    # # model.load_state_dict(torch.load('3018_model.pth'))
    #
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # params = list(uncertain_loss.parameters()) + list(model.parameters())
    # optimizer = optim.AdamW(params, lr=0.0001)
    #
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)
    #

    Config.expriment_id = 19_2
    writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    Config.model_name = "basic_4"
    Config.MODEL_SCALE = 4
    Config.IMG_WIDTH = 1536
    Config.IMG_HEIGHT = 512
    Config.FOCAL_ALPHA = 0.8
    Config.N_EPOCH = 10
    Config.MASK_WEIGHT = 100
    uncertain_loss = UncertaintyLoss().to(Config.device)
    Config.USE_MASK = False
    Config.FOUR_CHANNEL = True
    model = get_model(Config.model_name)
    model.load_state_dict(torch.load('19_model.pth'))

    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
                     uncertain_loss=uncertain_loss)
    predict(model)

    Config.expriment_id = 19_3
    writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    Config.model_name = "basic_4"
    Config.MODEL_SCALE = 4
    Config.IMG_WIDTH = 1536
    Config.IMG_HEIGHT = 512
    Config.FOCAL_ALPHA = 0.8
    Config.N_EPOCH = 10
    Config.MASK_WEIGHT = 1000
    uncertain_loss = UncertaintyLoss().to(Config.device)
    Config.USE_MASK = False
    Config.FOUR_CHANNEL = True
    model = get_model(Config.model_name)
    model.load_state_dict(torch.load('19_model.pth'))

    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
                     uncertain_loss=uncertain_loss)
    predict(model)