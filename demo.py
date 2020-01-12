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


train_lens = get_lens("pku-autonomous-driving/train.csv")
test_lens = get_lens("121_predictions.csv")
test_lens_new = get_lens("121_predictions_new.csv")
test_old = get_lens('/Users/tezign/Downloads/1221_predictions.csv')
