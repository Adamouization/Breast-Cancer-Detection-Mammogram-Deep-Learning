import os

import pandas as pd


def main() -> None:
    """
    Folder to adjust csv to remove images with multiple masks i.e. when they have multiple tumours
    """
    csv_mask_root_file = '../data/CBIS-DDSM-mask/training.csv'      
    csv_image_root_file = '../data/CBIS-DDSM/training.csv'   
    
    csv_mask_root_file_testing = '../data/CBIS-DDSM-mask/testing.csv'      
    csv_image_root_file_testing = '../data/CBIS-DDSM/testing.csv'

    csv_output_path = '../data/CBIS-DDSM-mask'       # csv output folder
    
    join_paths_and_remove_duplicates(csv_mask_root_file, csv_image_root_file, "shortened_mask_training",csv_output_path)
    join_paths_and_remove_duplicates(csv_mask_root_file_testing, csv_image_root_file_testing, "shortened_mask_testing",csv_output_path)

    

    
def join_paths_and_remove_duplicates(mask_path, img_path, file_name, output_path):
    df_masks = pd.read_csv(mask_path, usecols=["img","img_path","label"])
    df_masks['shortened_img_name'] = df_masks.apply(lambda x: x["img"][:-2], axis=1)
    df_masks.drop_duplicates(subset="shortened_img_name", keep = False, inplace=True)
    df_masks.rename(columns = {"img_path": "mask_img_path", "img": "mask_img"}, inplace=True)
    
    df_images = pd.read_csv(img_path, usecols=["img","img_path","label"])
    
    df = df_images.merge(df_masks, left_on=["img", "label"], right_on=["shortened_img_name", "label"] , how="inner")
    df = df[['img', 'img_path', 'mask_img_path', 'label']]
    
    df.to_csv(output_path + '/' + file_name + '.csv',
                             index=False)

if __name__ == '__main__':
    main()

