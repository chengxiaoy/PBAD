from efficientnet_pytorch import EfficientNet
import numpy as np  # linear algebra
from unet.unet_parts import *
from unet.unet_model import UNet
from config import Config


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


def get_model(model_name):
    if model_name == 'basic':
        return MyUNet(Config.N_CLASS).to(Config.device)
    if model_name == 'basic_unet':
        return UNet(3, Config.N_CLASS).to(Config.device)


if __name__ == '__main__':
    x = torch.randn(4, 3, 320, 1024)
    # model = get_model('basic')
    # y = model(x)
    # print(y.size())

    model = get_model('basic_unet')
    y = model(x)
    print(y.size())
