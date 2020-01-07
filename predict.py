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
    Config.expriment_id =11
    Config.USE_MASK = True
    Config.model_name = "basic_4"
    Config.MODEL_SCALE = 4

    model = get_model(Config.model_name)
    model.load_state_dict(torch.load('11_model.pth'))
    model.eval()
    predict(model)
