from img_preprocessing import str2coords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# "pku-autonomous-driving/train.csv"

def get_lens(file_path):
    train = pd.read_csv(file_path)
    train = train.fillna("")
    lens = [len(str2coords(s)) for s in train['PredictionString']]
    return lens


def show(lens):
    plt.figure(figsize=(15, 6))
    sns.countplot(lens)
    plt.xlabel('Number of cars in image')
    plt.show()


#
# train_lens = get_lens("pku-autonomous-driving/train.csv")
# test_lens = get_lens("121_predictions.csv")
# test_lens_new = get_lens("121_predictions_new.csv")
# test_old = get_lens('/Users/tezign/Downloads/1221_predictions.csv')

from torch.optim import lr_scheduler
from torch import optim
from torchvision.models import resnet18

optimizer = optim.Adam(resnet18(True).parameters(), lr=0.1, weight_decay=0.01)

# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
n_epoch = 40
lr_list = []
for i in range(n_epoch):
    optimizer.step()
    scheduler.step()
    lr_list.append(scheduler.get_lr())

plt.figure()
plt.plot(lr_list)
plt.show()
