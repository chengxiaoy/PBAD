from car_dataset import *
from network import *
from config import Config
import joblib


def predict(model, thr=0.0):
    predictions = []
    test_dataset = get_data_set()[2]
    test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=4)
    model.eval()

    outputs = []

    for img, _, _ in tqdm(test_loader):
        with torch.no_grad():
            output = model(img.to(Config.device))
        if Config.model_name.startswith('dla'):
            output = torch.cat((output[0]['mp'], output[0]['xyz'], output[0]['roll']), dim=1)

        if Config.model_name == 'hourglass':
            output = torch.cat((output[-1]['mp'], output[-1]['xyz'], output[-1]['roll']), dim=1)

        output = output.data.cpu().numpy()
        outputs.append(output)

        for out in output:
            coords = extract_coords(out, thr)
            s = coords2str(coords)
            predictions.append(s)

    joblib.dump(outputs, 'outputs.pkl')

    test = pd.read_csv(Config.DATA_PATH + 'sample_submission.csv')
    test['PredictionString'] = predictions
    test.to_csv(str(Config.expriment_id) + '_predictions.csv', index=False)


if __name__ == '__main__':
    Config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Config.expriment_id = 17_1
    Config.model_name = "basic_4"
    Config.MODEL_SCALE = 4
    Config.IMG_WIDTH = 1536
    Config.IMG_HEIGHT = 512
    Config.FOCAL_ALPHA = 0.9
    Config.N_EPOCH = 30
    Config.MASK_WEIGHT = 500
    Config.USE_MASK = True
    model = get_model(Config.model_name)
    model.load_state_dict(torch.load('171_model.pth'))

    predict(model, 0.0)
