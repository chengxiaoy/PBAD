from car_dataset import *
from network import *
from config import Config


def predict(model):
    predictions = []
    test_dataset = get_data_set()[2]
    test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=4)
    model.eval()

    for img, _, _ in tqdm(test_loader):
        with torch.no_grad():
            output = model(img.to(Config.device))
        if Config.model_name.startswith('dla'):
            output = torch.cat((output[0]['mask'], output[0]['xyz'], output[0]['roll']), dim=1)
        output = output.data.cpu().numpy()

        for out in output:
            coords = extract_coords(out)
            s = coords2str(coords)
            predictions.append(s)

    test = pd.read_csv(Config.DATA_PATH + 'sample_submission.csv')
    test['PredictionString'] = predictions
    test.to_csv(str(Config.expriment_id) + '_predictions.csv', index=False)

if __name__ == '__main__':



    Config.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    Config.expriment_id = 12_1
    Config.model_name = "basic_4"
    Config.MODEL_SCALE = 4
    Config.IMG_WIDTH = 1536
    Config.IMG_HEIGHT = 512
    Config.FOCAL_ALPHA = 0.75
    Config.N_EPOCH = 30
    Config.MASK_WEIGHT = 200
    Config.USE_UNCERTAIN_LOSS = True
    Config.USE_MASK = True
    model = get_model(Config.model_name)
    model.load_state_dict(torch.load('121_model.pth'))

    predict(model)
