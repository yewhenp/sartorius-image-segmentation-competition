import pandas as pd

# START PROJECT IMPORTS
from ..constants import WIDTH, HEIGHT
# END PROJECT_IMPORTS


def load_train_labels(path: str) -> pd.DataFrame:
    train_df = pd.read_csv(path)

    mean_width, mean_height = train_df["width"].mean(), train_df["height"].mean()
    assert int(mean_width) == WIDTH and int(mean_height) == HEIGHT, "Heights or Widths are not ok"

    train_df.drop(columns=[
        "width", "height", "cell_type", "plate_time", "sample_date", "sample_id", "elapsed_timedelta"
    ], inplace=True)

    grouped = train_df.groupby("id")
    return grouped
