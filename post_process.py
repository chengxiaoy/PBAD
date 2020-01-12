import numpy as np
import joblib
from car_dataset import *
from config import Config

def str2coords(s, names=['confi', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    '''
    Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    '''
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords


outputs = joblib.load("outputs.pkl")
predictions = []
thr = 0.3
for output in outputs:
    for out in output:
        coords = extract_coords(out, thr)
        s = coords2str(coords)
        predictions.append(s)

test = pd.read_csv(Config.DATA_PATH + 'sample_submission.csv')
test['PredictionString'] = predictions
test.to_csv(str(Config.expriment_id) + '_predictions.csv', index=False)