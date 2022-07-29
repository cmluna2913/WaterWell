import configparser
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import datetime as dt
import os
# import warnings
# warnings.filterwarnings("ignore")

import numpy as np

# For vizualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix

# To create a quick model to look at Feature Importances
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# To save csv's with current date information
import datetime as dt

def simple_clean(train_df, test_df, to_drop):
    '''
    Set 'id' as the index and drop specified columns.
    '''
    train_df.index, test_df.index = train_df['id'], test_df['id']
    train_df.drop(columns=to_drop, inplace=True)
    test_df.drop(columns=to_drop, inplace=True)

def fill_nulls(df, fill_dict):
    '''
    Add fill value to columns with missing data if not included already.
    Then, fill the df with the corresponding fill value from the fill dictionary.
    '''
    for col in ['funder', 'installer', 'subvillage', 'scheme_management', 'scheme_name']:
        try:
                df[col] = df[col].cat.add_categories(fill_dict[col])
        except:
            continue
    df.fillna(fill_dict, inplace=True)

def transform_data(train_df, test_df, n=20):
    '''
    Separates categorical columns and one hot encode columns.
    Return concatenation of numerical columns and encoded columns.
    '''
    trn_cats = train_df.select_dtypes(include='category')
    tst_cats = test_df.select_dtypes(include='category')
    trn_nums = train_df.select_dtypes(exclude='category')
    tst_nums = test_df.select_dtypes(exclude='category')

    trn_cats, test_cats, cat_dict=lower_features(trn_cats, tst_cats, n)
    trn_cats, tst_cats = ohe_me(trn_cats, tst_cats)

    train_df = trn_nums.merge(trn_cats, left_index=True, right_index=True, how='inner')
    test_df = tst_nums.merge(tst_cats, left_index=True, right_index=True, how='inner')

    return train_df, test_df, cat_dict

def ohe_me(trn_cats, tst_cats):
    '''
    One hot encode columns in both training and testing set. 
    Both are done at the same time to ensure handling of unknown 
    categorical values.
    '''
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(trn_cats)
    trn_ohe = pd.DataFrame(ohe.transform(trn_cats).toarray(), 
                            columns = ohe.get_feature_names(), index=trn_cats.index)
    tst_ohe = pd.DataFrame(ohe.transform(tst_cats).toarray(), 
                            columns = ohe.get_feature_names(), index = tst_cats.index)
    return trn_ohe, tst_ohe

def lower_features(trn_cats, tst_cats, n):
    '''
    Lower categories in columns to the specified number of categories (default=20).
    For the specified columns, it will take the top n categories and re-categorize
    remaining categories to the specified value.
    This step is done to reduce dimensionality.
    '''
    cols = trn_cats.columns
    cat_dict = {}
    for col in cols:
        try:
            temp = trn_cats[col].value_counts().head(n).keys()
            trn_cats[col] = trn_cats[col].apply(lambda x: 'Other' if x not in temp else x)
            tst_cats[col] = tst_cats[col].apply(lambda x: 'Other' if x not in temp else x)
            cat_dict[col] = list(trn_cats[col].cat.categories)
        except:
            continue
    trn_cats[cols] = trn_cats[cols].astype('category')
    tst_cats[cols] = tst_cats[cols].astype('category')
    return trn_cats, tst_cats, cat_dict

def save_cleaned_data(train_df, test_df, current_time, temp_output):
    '''
    Saves cleaned training and testing set as .pkl files with the current
    date and time. Date and time is in the format of DDMMYY_HHMM, with hours
    and minutes including AM or PM.
    '''
    try:
        os.makedirs(temp_output)
    except:
        pass
    train_df.to_pickle(temp_output+'train_data.pkl')
    test_df.to_pickle(temp_output+'test_data.pkl')

def make_notes(current_time, temp_output, dtype_dict, to_drop, fill_dict, n, cat_dict):
    notes = f"""Associated date and time: {current_time}
Files saved to {temp_output}\n\n
dtype_dict:
{improve_format_dct(dtype_dict)}\n
drop_list:
{improve_format_lst(to_drop)}\n
fill_dict:
{improve_format_dct(fill_dict)}\n
Grabbed top {n} features\n\n
Categorical columns and associated categories:
{improve_format_dct(cat_dict)}
"""
    with open(f"{temp_output}experiment_notes.txt", 'w') as f:
        f.write(notes)

def improve_format_lst(my_list):
    temp = ""
    for l in my_list:
        temp+=f"\t{l},\n"
    return temp[:-1]

def improve_format_dct(my_dict):
    temp = ""
    for k,v in my_dict.items():
        temp += f"\t{k}:{v},\n"
    return temp[:-1]

def import_me(data_path, dtype_dict=None):
    if ".csv" in data_path:
        df = pd.read_csv(data_path, dtype=dtype_dict)
    else:
        df = pd.read_pkl(data_path)
    return df

def get_cleaned_sets(train_df, test_df, to_drop, output, fill_dict, dtype_dict=None, n=20,return_output=False):
    if isinstance(train_df, str):
        train_df = import_me(train_df, dtype_dict)
    if isinstance(test_df, str):
        test_df = import_me(test_df, dtype_dict)

    current_time = dt.datetime.now().strftime("%d%m%y_%I%M%p")
    temp_output = f"{output}experiments/{current_time}/"

    simple_clean(train_df, test_df, to_drop)
    fill_nulls(train_df, fill_dict)
    fill_nulls(test_df, fill_dict)
    train_df, test_df, cat_dict = transform_data(train_df, test_df)

    save_cleaned_data(train_df, test_df,current_time, temp_output)
    make_notes(current_time, temp_output,
            dtype_dict, to_drop, fill_dict, n, cat_dict)

    print(f"Cleaning successful.\nAssociated time is {current_time}")

    if return_output:
        return train_df, test_df, temp_output