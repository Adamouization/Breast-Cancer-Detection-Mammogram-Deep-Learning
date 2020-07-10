import os

import pandas as pd


def main() -> None:
    """
    Initial dataset pre-processing for the CBIS-DDSM dataset:
        * Retrieves the path of all images and filters out the cropped images.
        * Imports original CSV files with the full image information (patient_id, left or right breast, pathology,
        image file path, etc.).
        * Filters out the cases with more than one pathology in the original csv files.
        * Merges image path which extracted on GPU machine and image pathology which is in the original csv file on image id
        * Outputs 4 CSV files.

        Generate CSV file columns:
          img: image id (e.g Calc-Test_P_00038_LEFT_CC => <case type>_<patient id>_<left or right breast>_<CC or MLO>)
          img_path: image path on the GPU machine
          label: image pathology (BENIGN or MALIGNANT)
    :return: None
    """
    csv_root = '../data/CBIS-DDSM/ori_csv'      # original csv folder
    img_root = '/cs/tmp/datasets/CBIS-DDSM'     # dataset folder
    csv_output_path = '../data/CBIS-DDSM'       # csv output folder

    folders = os.listdir(img_root)

    cases_dict = dict()  # save image id and path

    for f in folders:
        if f.endswith('_CC') or f.endswith('_MLO'):  # filter out the cropped images
            path = list()

            for root, dirs, files in os.walk(img_root + '/' + f):  # retrieve the path of image
                for d in dirs:
                    path.append(d)
                for filename in files:
                    path.append(filename)

            img_path = img_root + '/' + f + '/' + '/'.join(path)  # generate image path
            cases_dict[f] = img_path

    df = pd.DataFrame(list(cases_dict.items()), columns=['img', 'img_path'])  # transform image dictionary to dataframe

    # image type and csv file name mapping
    type_dict = {'Calc-Test': 'calc_case_description_test_set.csv',
                 'Calc-Training': 'calc_case_description_train_set.csv',
                 'Mass-Test': 'mass_case_description_test_set.csv',
                 'Mass-Training': 'mass_case_description_train_set.csv'}

    for t in type_dict.keys():  # handle images based on the type
        df_subset = df[df['img'].str.startswith(t)]

        df_csv = pd.read_csv(csv_root + '/' + type_dict[t],
                             usecols=['pathology', 'image file path'])  # read original csv file
        df_csv['img'] = df_csv.apply(lambda row: row['image file path'].split('/')[0],
                                     axis=1)  # extract image id from the path
        df_csv['pathology'] = df_csv.apply(
            lambda row: 'BENIGN' if row['pathology'] == 'BENIGN_WITHOUT_CALLBACK' else row['pathology'],
            axis=1)  # replace pathology 'BENIGN_WITHOUT_CALLBACK' to 'BENIGN'

        df_cnt = pd.DataFrame(
            df_csv.groupby(['img'])['pathology'].nunique()).reset_index()  # count dictict pathology for each image id
        df_csv = df_csv[~df_csv['img'].isin(
            list(df_cnt[df_cnt['pathology'] != 1]['img']))]  # filter out the image with more than one pathology
        df_csv = df_csv.drop(columns=['image file path'])
        df_csv = df_csv.drop_duplicates(
            keep='first')  # remove duplicate data (because original csv list all abnormality area, that make one image id may have more than one record)

        df_subset_new = pd.merge(df_subset, df_csv, how='inner',
                                 on='img')  # merge image path and image pathology on image id
        df_subset_new = df_subset_new.rename(columns={'pathology': 'label'})  # rename column 'pathology' to 'label'
        df_subset_new.to_csv(csv_output_path + '/' + t.lower() + '.csv',
                             index=False)  # output merged dataframe in csv format

        print(t)
        print('data_cnt: %d' % len(df_subset_new))
        print('multi pathology case:')
        print(list(df_cnt[df_cnt['pathology'] != 1]['img']))
        print()

    print('Finished pre-processing CSV for the CBIS-DDSM dataset.')


if __name__ == '__main__':
    main()

# ----------------------------------

# Calc-Test
# data_cnt: 282
# multi pathology case:
# ['Calc-Test_P_00353_LEFT_CC', 'Calc-Test_P_00353_LEFT_MLO']

# Calc-Training
# data_cnt: 1220
# multi pathology case:
# ['Calc-Training_P_00600_LEFT_CC', 'Calc-Training_P_00600_LEFT_MLO', 'Calc-Training_P_00937_RIGHT_CC', 'Calc-Training_P_00937_RIGHT_MLO', 'Calc-Training_P_01284_RIGHT_MLO', 'Calc-Training_P_01819_LEFT_CC', 'Calc-Training_P_01819_LEFT_MLO']

# Mass-Test
# data_cnt: 359
# multi pathology case:
# ['Mass-Test_P_00969_LEFT_CC', 'Mass-Test_P_00969_LEFT_MLO']

# Mass-Training
# data_cnt: 1225
# multi pathology case:
# ['Mass-Training_P_00419_LEFT_CC', 'Mass-Training_P_00419_LEFT_MLO', 'Mass-Training_P_00797_LEFT_CC', 'Mass-Training_P_00797_LEFT_MLO', 'Mass-Training_P_01103_RIGHT_CC', 'Mass-Training_P_01103_RIGHT_MLO']
