import numpy as np
import joblib
from config import Config
import pandas as pd

def str2coords(s, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    '''
    Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    '''
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coord = dict(zip(names, l.astype('float')))
        if coord['confidence'] > 0.58:
            coords.append(coord)
    return coords


def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return ' '.join(s)


predict = pd.read_csv('1221_predictions.csv')
predictions = []
for ss in predict['PredictionString']:
    if ss is np.nan or ss is '':
        predictions.append('')
    else:
        coords = str2coords(ss)
        predictions.append(coords2str(coords))




test = pd.read_csv(Config.DATA_PATH + 'sample_submission.csv')
test['PredictionString'] = predictions
test.to_csv(str(Config.expriment_id) + '_predictions.csv', index=False)
