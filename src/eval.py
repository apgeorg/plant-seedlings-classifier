import utils
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model

img_width, img_height = (299, 299)
modelname = 'inceptResNetV2'

def eval():
    # Get classes
    lb = LabelBinarizer()
    lbenc = lb.fit(utils.get_classes())

    # Load model
    model = load_model('../models/' + 'inceptResNetV2-03-0.410.h5')

    # Get test data
    X_test, X_test_id = utils.get_test('../input/test', img_width, img_height)

    # Predict on test data
    preds = model.predict(X_test, verbose=1)

    # Create ensembling file
    df_csv = utils.to_csv_ens(lbenc.inverse_transform(preds), preds, X_test_id, utils.get_classes(),
                              output_path="../submissions/", filename=modelname)
    # Create submission file
    subm = utils.create_submission(lbenc.inverse_transform(preds), X_test_id, output_path="../submissions/",
                                   filename=modelname, isSubmission=True)

if __name__ == "__main__":
    eval()