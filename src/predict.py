import os
import pandas as pd

from typing import Dict

# START PROJECT IMPORTS
from .data_processing.data_generator import SubmissionDataLoader
from .data_processing.post_processing import mask2rle
from .models import get_model
from .constants import REDUCED_HEIGHT, WIDTH, ConfigKeys as ck
from .visualisation import display
# END PROJECT_IMPORTS


def predict_submission(cnf: Dict, weights_path: str = None) -> None:
    print("Loading weights...")
    model = get_model(cnf, input_shape=(REDUCED_HEIGHT, WIDTH, 3))
    if weights_path is None:
        weights_path = os.path.join(cnf[ck.WEIGHTS_DIR], cnf[ck.EXPERIMENT_NAME], "val_best.h5")
    model.load_weights(weights_path)

    print("Submision photos predicting...")
    submission_dl = SubmissionDataLoader(cnf)
    data_generator_submission = submission_dl.load_data()
    submission = pd.DataFrame([], columns=['id', 'predicted'])
    for i in range(len(data_generator_submission)):
        y_hat = model.predict(data_generator_submission[i][0])
        image_id = submission_dl.df.iloc[i]['id']
        mask_df = mask2rle(image_id, y_hat[0])
        submission = pd.concat([submission, mask_df])
        # if cnf[ck.DISPLAY]:
        #     display([data_generator_submission[i][0][0], data_generator_submission[i][1][0], y_hat[0]])

    submission.to_csv("submission.csv", index=False)
