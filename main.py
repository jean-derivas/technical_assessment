import os
import sqlite3
from sqlite3 import Connection

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def get_image_from_database(
        depth_min: float,
        depth_max: float,
        connection: Connection
) -> pd.DataFrame:
    """Get the image from the database where it is stored

    Args:
        depth_min: the depth minimum of the image to retrieve
        depth_max: the depth maximum of the image to retrieve
        connection: the connection to access the database

    Returns:
        the dataframe containing data from database
    """
    if depth_min >= depth_max:
        raise ValueError(
            f"depth_min ({depth_min}) should be smaller than depth_max"
            f"({depth_max})"
        )
    return pd.read_sql(
        "select * "
        "from image "
        f"where depth >= {depth_min} and depth <= {depth_max}",
        connection
    ).set_index("depth")


# Press the green button in the gutter to run the script.
def main():
    cleaning()

    df = load_data_from_file()
    array = df.to_numpy()

    resized = cv2.resize(array, (150, array.shape[0]))

    df_resized = pd.DataFrame(
        resized, columns=[f'col_{x}' for x in range(1, 151)],
        index=df.index
    )

    con = sqlite3.connect("image.db")
    df_resized.to_sql(name="image", con=con)

    # get only a part of the image to show the API is working
    df_filtered = get_image_from_database(9100, 9300, con)
    filtered = df_filtered.to_numpy()

    cm = plt.get_cmap('gist_rainbow')

    colored_image = cm(filtered / 255)

    export_colored_image(colored_image, 'colored_filtered.png')


def cleaning():
    """used in case of multiple relaunch to avoid error"""
    try:
        os.remove("image.db")
    except FileNotFoundError:
        pass


def export_colored_image(colored_image: np.ndarray, filename: str):
    Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(filename)


def load_data_from_file() -> pd.DataFrame:
    df = pd.read_csv("img.csv").set_index("depth").dropna()
    return df


if __name__ == '__main__':
    main()
