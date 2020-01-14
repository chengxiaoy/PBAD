from efficientnet_pytorch import EfficientNet
import numpy as np  # linear algebra
from unet.unet_parts import *
from unet.unet_model import UNet, UNet_EFF
from config import Config
from models.networks.dlav0 import get_pose_net, dla34, dla102x
from img_preprocessing import imread
from models.networks.hourglass import get_large_hourglass_net


def get_mesh(batch_size, shape_x, shape_y):
    mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
    mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mesh = torch.cat([torch.tensor(mg_x).to(Config.device), torch.tensor(mg_y).to(Config.device)], 1)
    return mesh


class MyUNet(nn.Module):
    '''Mixture of previous classes'''

    def __init__(self, n_classes):
        super(MyUNet, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')

        self.conv0 = DoubleConv(3, 64)
        self.conv1 = DoubleConv(64, 128)
        self.conv2 = DoubleConv(128, 512)
        self.conv3 = DoubleConv(512, 1024)

        self.mp = nn.MaxPool2d(2)

        self.up1 = Up(1280 + 1024, 512)
        self.up2 = Up(512 + 512, 256)
        self.outc = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
        # x0 = torch.cat([x, mesh1], 1)
        x0 = x
        x1 = self.mp(self.conv0(x0))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))

        x_center = x[:, :, :, Config.IMG_WIDTH // 8: -Config.IMG_WIDTH // 8]
        feats = self.base_model.extract_features(x_center)
        bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8]).to(Config.device)
        feats = torch.cat([bg, feats, bg], 3)

        # Add positional info
        mesh2 = get_mesh(batch_size, feats.shape[2], feats.shape[3])
        # feats = torch.cat([feats, mesh2], 1)

        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        x = self.outc(x)
        return x



class MyUNet_7(nn.Module):
    '''Mixture of previous classes'''

    def __init__(self, n_classes):
        super(MyUNet_7, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b7')

        self.conv0 = DoubleConv(3, 64)
        self.conv1 = DoubleConv(64, 128)
        self.conv2 = DoubleConv(128, 512)
        self.conv3 = DoubleConv(512, 1024)

        self.mp = nn.MaxPool2d(2)

        self.up1 = Up(3584, 512)
        self.up2 = Up(512 + 512, 256)
        self.outc = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
        # x0 = torch.cat([x, mesh1], 1)
        x0 = x
        x1 = self.mp(self.conv0(x0))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))

        x_center = x[:, :, :, Config.IMG_WIDTH // 8: -Config.IMG_WIDTH // 8]
        feats = self.base_model.extract_features(x_center)
        bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8]).to(Config.device)
        feats = torch.cat([bg, feats, bg], 3)

        # Add positional info
        mesh2 = get_mesh(batch_size, feats.shape[2], feats.shape[3])
        # feats = torch.cat([feats, mesh2], 1)

        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        x = self.outc(x)
        return x


class MyUNet4_V2(nn.Module):
    '''Mixture of previous classes'''

    def __init__(self, base_model, input_channels, n_classes):
        super(MyUNet4_V2, self).__init__()
        if base_model == 'dla34':
            self.base_model = dla34(pretrained=True, return_levels=True)
        elif base_model == 'dla102x':
            self.base_model = dla102x(pretrained=True, return_levels=True)

        self.conv0 = DoubleConv(input_channels, 64)
        self.conv1 = DoubleConv(64, 128)
        self.conv2 = DoubleConv(128, 512)
        self.conv3 = DoubleConv(512, 1024)
        self.mp = nn.MaxPool2d(2)

        if base_model == 'dla34':
            self.up1 = Up(512 + 1024, 512)
        elif base_model == 'dla34':
            self.up1 = Up(1024 + 1024, 512)
        self.up2 = Up(512 + 512, 256)
        self.up3 = Up(128 + 256, 256)
        self.outc = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
        # x0 = torch.cat([x, mesh1], 1)
        x0 = x
        x1 = self.mp(self.conv0(x0))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))

        x_center = x[:, :3, :, Config.IMG_WIDTH // 8: -Config.IMG_WIDTH // 8]
        feats = self.base_model(x_center)[5]
        bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8]).to(Config.device)
        feats = torch.cat([bg, feats, bg], 3)

        # Add positional info
        mesh2 = get_mesh(batch_size, feats.shape[2], feats.shape[3])
        # feats = torch.cat([feats, mesh2], 1)

        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.outc(x)
        return x


class MyUNet4(nn.Module):
    '''Mixture of previous classes'''

    def __init__(self, input_channels, n_classes):
        super(MyUNet4, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')

        self.conv0 = DoubleConv(input_channels, 64)
        self.conv1 = DoubleConv(64, 128)
        self.conv2 = DoubleConv(128, 512)
        self.conv3 = DoubleConv(512, 1024)

        self.mp = nn.MaxPool2d(2)

        if Config.model_name.__contains__('mesh'):
            self.up1 = Up(1280 + 1024 + 2, 512)
        else:
            self.up1 = Up(1280 + 1024, 512)
        self.up2 = Up(512 + 512, 256)
        self.up3 = Up(128 + 256, 256)
        self.outc = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        if Config.model_name.__contains__('mesh'):
            mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
            x0 = torch.cat([x, mesh1], 1)
        else:
            x0 = x
        x1 = self.mp(self.conv0(x0))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))

        x_center = x[:, :3, :, Config.IMG_WIDTH // 8: -Config.IMG_WIDTH // 8]
        feats = self.base_model.extract_features(x_center)
        bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8]).to(Config.device)
        feats = torch.cat([bg, feats, bg], 3)

        # Add positional info
        if Config.model_name.__contains__('mesh'):
            mesh2 = get_mesh(batch_size, feats.shape[2], feats.shape[3])
            feats = torch.cat([feats, mesh2], 1)

        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.outc(x)
        return x


def get_model(model_name):
    if model_name == 'basic':
        model = MyUNet(Config.N_CLASS)

    if model_name == 'basic_b7':
        model = MyUNet_7(Config.N_CLASS)

    if model_name == "basic_4_mesh":
        model = MyUNet4(5, Config.N_CLASS)
    if model_name == 'basic_4':
        if not Config.FOUR_CHANNEL:
            model = MyUNet4(3, Config.N_CLASS)
        else:
            model = MyUNet4(4, Config.N_CLASS)

    if model_name == "basic_4_dla_34":
        if not Config.FOUR_CHANNEL:
            model = MyUNet4_V2('dla34', 3, Config.N_CLASS)
        else:
            model = MyUNet4_V2('dla34', 4, Config.N_CLASS)

    if model_name == "basic_4_dla_102x":
        if not Config.FOUR_CHANNEL:
            model = MyUNet4_V2('dla102x', 3, Config.N_CLASS)
        else:
            model = MyUNet4_V2('dla102x', 4, Config.N_CLASS)

    if model_name == 'basic_unet':
        if not Config.FOUR_CHANNEL:
            model = UNet(3, Config.N_CLASS)
        else:
            model = UNet(4, Config.N_CLASS)
    if model_name == 'unet':
        model = UNet_EFF("efficientnet-b0", 8)

    if model_name == 'unet_7':
        model = UNet_EFF("efficientnet-b7", 8)
    # if model_name == 'dla34':
    #     model = get_pose_net(34, {"mask": 1, "regr": 7})
    if model_name == 'dla34_2':
        model = get_pose_net(34, {"mp": 1, "xyz": 3, "roll": 4})
    if model_name == 'hourglass':
        model = get_large_hourglass_net(None, {"mp": 1, "xyz": 3, "roll": 4}, None)
    if model_name == 'dla102_x':
        model = get_pose_net("102x", {"mp": 1, "xyz": 3, "roll": 4})
    if Config.PARALLEL and str(Config.device) != 'cpu':
        model = torch.nn.DataParallel(model, device_ids=Config.device_ids)
    model = model.to(Config.device)
    return model


if __name__ == '__main__':
    x = torch.randn(4, 3, 512, 1536)
    # model = get_model('basic')
    # y = model(x)
    # print(y.size())

    model = get_model('hourglass')
    y = model(x)
    print(y.size())
