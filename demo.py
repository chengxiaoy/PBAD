from img_preprocessing import str2coords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("pku-autonomous-driving/train.csv")
lens = [len(str2coords(s)) for s in train['PredictionString']]

plt.figure(figsize=(15, 6))
sns.countplot(lens)
plt.xlabel('Number of cars in image')
plt.show()


valid = pd.read_csv("/Users/tezign/Downloads/11_predictions.csv")
valid = valid.fillna("")
lens = [len(str2coords(s)) for s in valid['PredictionString']]

plt.figure(figsize=(15, 6))
sns.countplot(lens)
plt.xlabel('Number of cars in image')
plt.show()