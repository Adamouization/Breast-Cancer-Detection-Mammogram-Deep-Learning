import os

import cv2
import pandas as pd


def main() -> None:
    """
    Initial dataset pre-processing for the mini-MIAS dataset:
        * Imports a CSV file with the image names and their label
        * Cleans the labels column by replacing empty cells with 'N', corresponding to normal cases.
        * Organises all images into labelled directories instead of one large directory with all images from all
        classes.
        * Converts all PGM images to PNG format.
    :return: None
    """
    KEY_LABEL = 3
    df = pd.read_csv("../data/mini-MIAS/data_description.csv", header=None, index_col=0)
    df[KEY_LABEL].fillna('N', inplace=True)  # Empty values correspond to normal cases ('M'=malignant, 'B'=benign).
    df[3].str.strip()  # Strip leading and trailing spaces in label column.

    for img_pgm in os.listdir("../data/mini-MIAS/images_original"):
        if img_pgm.endswith(".pgm"):
            img = cv2.imread("../data/mini-MIAS/images_original/{}".format(img_pgm))
            img_name = img_pgm.split(".")[0]
            label = df.loc[img_name].loc[3]
            if label == 'N':
                new_path = "../data/mini-MIAS/images_processed/normal_cases/{}.png".format(img_name)
            elif label == 'B ':
                new_path = "../data/mini-MIAS/images_processed/benign_cases/{}.png".format(img_name)
            elif label == 'M ':
                new_path = "../data/mini-MIAS/images_processed/malignant_cases/{}.png".format(img_name)
            cv2.imwrite(new_path, img)
            print("Converted {} from PGM to PNG ({} case).".format(img_pgm, label))
    print("Finished converting and sorting dataset.")


if __name__ == '__main__':
    main()
