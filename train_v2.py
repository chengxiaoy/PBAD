from training import *

# Gets the GPU if there is one, otherwise the cpu


if __name__ == '__main__':
    Config.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Config.expriment_id = 3
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # model = get_model(Config.model_name)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer)
    # predict(model)
    #
    # Config.expriment_id = 4
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic"
    # Config.FOCAL_ALPHA = 0.25
    # model = get_model(Config.model_name)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer)
    # predict(model)

    # Config.expriment_id = 6
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic"
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 30
    # Config.USE_MASK = True
    # model = get_model(Config.model_name)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer)
    # predict(model)

    # Config.expriment_id = 8
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic"
    # Config.IMG_WIDTH = 1536
    # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 10
    # Config.MASK_WEIGHT = 200
    # Config.USE_MASK = True
    # model = get_model(Config.model_name)
    # model.load_state_dict(torch.load('8_model.pth'))
    # optimizer = optim.Adam(model.parameters(), lr=0.00001)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer)
    # predict(model)

    # Config.expriment_id = 10_2
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
    # model = get_model(Config.model_name)
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # params = list(uncertain_loss.parameters()) + list(model.parameters())
    # optimizer = optim.AdamW(params, lr=0.001, weight_decay=0.01)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)

    # Config.expriment_id = 10_3
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic_4"
    # Config.MODEL_SCALE = 4
    # # Config.IMG_WIDTH = 1536
    # # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 10
    # Config.MASK_WEIGHT = 20
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
    #
    #
    # Config.expriment_id = 10_4
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic_4"
    # Config.MODEL_SCALE = 4
    # # Config.IMG_WIDTH = 1536
    # # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 10
    # Config.MASK_WEIGHT = 1000
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

    # Config.expriment_id = 10_5
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic_4"
    # Config.MODEL_SCALE = 4
    # # Config.IMG_WIDTH = 1536
    # # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 10
    # Config.MASK_WEIGHT = 200
    # Config.USE_UNCERTAIN_LOSS = True
    # Config.USE_MASK = False
    # Config.FOUR_CHANNEL = True
    # model = get_model(Config.model_name)
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # params = list(uncertain_loss.parameters()) + list(model.parameters())
    # optimizer = optim.AdamW(params, lr=0.001, weight_decay=0.01)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)

    # Config.expriment_id = 10_51
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic_4"
    # Config.MODEL_SCALE = 4
    # # Config.IMG_WIDTH = 1536
    # # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 10
    # Config.MASK_WEIGHT = 200
    # Config.USE_UNCERTAIN_LOSS = True
    # Config.USE_MASK = False
    # Config.FOUR_CHANNEL = True
    # model = get_model(Config.model_name)
    # model.load_state_dict(torch.load('105_model.pth'))
    #
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # params = list(uncertain_loss.parameters()) + list(model.parameters())
    # optimizer = optim.AdamW(params, lr=0.0001, weight_decay=0.01)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)

    # Config.expriment_id = 10_6
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic_4_dla_34"
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
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)

    # Config.expriment_id = 10_7
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic_4_dla_102x"
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
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)

    # Config.expriment_id = 10_8
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic_4_mesh"
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
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[12, 20], gamma=0.1)
    #
    # model = training(model, optimizer, scheduler=scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)

    # Config.expriment_id = 11
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic_4"
    # Config.MODEL_SCALE = 4
    # # Config.IMG_WIDTH = 1536
    # # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 10
    # Config.MASK_WEIGHT = 200
    # Config.USE_UNCERTAIN_LOSS = False
    #
    # Config.USE_MASK = True
    # model = get_model(Config.model_name)
    # model.load_state_dict(torch.load('11_model.pth'))
    # optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=None)
    # predict(model)

    # Config.expriment_id = 12
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic_4"
    # Config.MODEL_SCALE = 4
    # Config.IMG_WIDTH = 1536
    # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 30
    # Config.MASK_WEIGHT = 200
    #
    # Config.USE_MASK = True
    # model = get_model(Config.model_name)
    # optimizer = optim.AdamW(model.parameters(), lr=0.001,weight_decay=0.01)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer)
    # predict(model)

    # Config.expriment_id = 12_1
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic_4"
    # Config.MODEL_SCALE = 4
    # Config.IMG_WIDTH = 1536
    # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 10
    # Config.MASK_WEIGHT = 500
    # Config.USE_UNCERTAIN_LOSS = True
    # Config.USE_MASK = True
    # model = get_model(Config.model_name)
    # model.load_state_dict(torch.load('12_model.pth'))
    #
    #
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # params = list(uncertain_loss.parameters()) + list(model.parameters())
    # optimizer = optim.AdamW(params, lr=0.0001, weight_decay=0.01)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)

    # Config.expriment_id = 12_2
    #     # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    #     # Config.model_name = "basic_4"
    #     # Config.MODEL_SCALE = 4
    #     # Config.IMG_WIDTH = 1536
    #     # Config.IMG_HEIGHT = 512
    #     # Config.FOCAL_ALPHA = 0.9
    #     # Config.N_EPOCH = 30
    #     # Config.MASK_WEIGHT = 500
    #     # uncertain_loss = UncertaintyLoss().to(Config.device)
    #     # Config.USE_MASK = True
    #     # model = get_model(Config.model_name)
    #     # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    #     # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    #     # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    #     # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #     #                  uncertain_loss=uncertain_loss)
    #     # predict(model)

    # Config.expriment_id = 13
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic_4"
    # Config.MODEL_SCALE = 4
    # Config.IMG_WIDTH = 2304
    # Config.IMG_HEIGHT = 768
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 30
    # Config.MASK_WEIGHT = 500
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

    # Config.expriment_id = 9
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic"
    # Config.IMG_WIDTH = 1536
    # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 10
    # Config.MASK_WEIGHT = 1000
    # Config.USE_MASK = True
    # model = get_model(Config.model_name)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer)
    # predict(model)

    # Config.expriment_id = 10
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic_4"
    # Config.MODEL_SCALE = 4
    # Config.IMG_WIDTH = 1536
    # Config.IMG_HEIGHT = 512
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 30
    # Config.USE_MASK = True
    # model = get_model(Config.model_name)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer)
    # predict(model)

    #
    #
    # Config.expriment_id = 30_181
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
    # model.load_state_dict(torch.load('3018_model.pth'))
    #
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # params = list(uncertain_loss.parameters()) + list(model.parameters())
    # optimizer = optim.AdamW(params, lr=0.001)
    #
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)
    #
    #
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
    # Config.N_EPOCH = 10
    # Config.MASK_WEIGHT = 10
    # Config.USE_UNCERTAIN_LOSS = True
    # Config.USE_MASK = True
    # Config.USE_GAUSSIAN = True
    # model = get_model(Config.model_name)
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # params = list(uncertain_loss.parameters()) + list(model.parameters())
    # optimizer = optim.AdamW(params, lr=0.001)
    #
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)

    # Config.expriment_id = 18
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic_b7"
    # Config.MODEL_SCALE = 8
    # Config.IMG_WIDTH = 1024
    # Config.IMG_HEIGHT = 320
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 10
    # Config.MASK_WEIGHT = 500
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # Config.USE_MASK = True
    # model = get_model(Config.model_name)
    # # model.load_state_dict(torch.load('17_model.pth'))
    #
    # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)


    # Config.expriment_id = 18_2
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic_b7"
    # Config.MODEL_SCALE = 8
    # Config.IMG_WIDTH = 1024
    # Config.IMG_HEIGHT = 320
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 20
    # Config.MASK_WEIGHT = 500
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # Config.USE_MASK = True
    # model = get_model(Config.model_name)
    # model.load_state_dict(torch.load('18_model.pth'))
    #
    # optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)

    # Config.expriment_id = 18_3
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic_b7"
    # Config.MODEL_SCALE = 8
    # Config.IMG_WIDTH = 1024
    # Config.IMG_HEIGHT = 320
    # Config.FOCAL_ALPHA = 0.75
    # Config.N_EPOCH = 20
    # Config.MASK_WEIGHT = 500
    # uncertain_loss = UncertaintyLoss().to(Config.device)
    # Config.USE_MASK = True
    # model = get_model(Config.model_name)
    # model.load_state_dict(torch.load('18_model.pth'))
    #
    # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)


    # Config.expriment_id = 30_183
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
    # Config.MASK_WEIGHT = 10
    # Config.USE_UNCERTAIN_LOSS = False
    # Config.USE_MASK = True
    # Config.USE_GAUSSIAN = True
    # model = get_model(Config.model_name)
    # model.load_state_dict(torch.load('3018_model.pth'))
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

    # Config.expriment_id = 10_25
    # writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    # Config.model_name = "basic_4"
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
    # optimizer = optim.AdamW(params, lr=0.001, weight_decay=0.01)
    # # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    # model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
    #                  uncertain_loss=uncertain_loss)
    # predict(model)


    Config.expriment_id = 28
    writer = SummaryWriter(logdir=os.path.join("board/", str(Config.expriment_id)))
    Config.model_name = "basic_b7"
    Config.MODEL_SCALE = 8
    Config.IMG_WIDTH = 1024
    Config.IMG_HEIGHT = 320
    Config.FOCAL_ALPHA = 0.75
    Config.N_EPOCH = 30
    Config.MASK_WEIGHT = 500
    uncertain_loss = UncertaintyLoss().to(Config.device)
    Config.USE_MASK = True
    model = get_model(Config.model_name)
    # model.load_state_dict(torch.load('17_model.pth'))

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=Config.N_EPOCH * len(train_loader) // 3, gamma=0.1)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    model = training(model, optimizer, scheduler=lr_scheduler, n_epoch=Config.N_EPOCH, writer=writer,
                     uncertain_loss=uncertain_loss)
    predict(model)




