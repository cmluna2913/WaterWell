import configparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import datetime as dt

config = configparser.ConfigParser()
config.read("../src/config.ini")

output = config['paths']['data_path']

train_lbls = pd.read_csv(config['paths']['train_labels'])
train_df = pd.read_csv(config['paths']['train_data'])
test_df = pd.read_csv(config['paths']['test_data'])
sub_form = pd.read_csv(config['paths']['sub_form'])

def get_cleaned_sets(trn_df_path, trn_lbls_path, tst_df_path, sub_form_path):
    dtype_dict = {'amount_tsh': 'float32',
                'funder': 'category',
                'gps_height': 'int16',
                'installer': 'category',
                'longitude': 'float16',
                'latitude': 'float16',
                'wpt_name': 'category',
                'num_private': 'int16',
                'basin': 'category',
                'subvillage': 'category',
                'region': 'category',
                'region_code': 'int8',
                'district_code': 'int8',
                'lga': 'category',
                'ward': 'category',
                'population': 'int16',
                'recorded_by': 'category',
                'scheme_management': 'category',
                'construction_year': 'int16',
                'extraction_type': 'category',
                'extraction_type_group': 'category',
                'extraction_type_class': 'category',
                'management': 'category',
                'management_group': 'category',
                'payment': 'category',
                'payment_type': 'category',
                'water_quality': 'category',
                'quality_group': 'category',
                'quantity': 'category',
                'quantity_group': 'category',
                'source': 'category',
                'source_type': 'category',
                'source_class': 'category',
                'waterpoint_type': 'category',
                'waterpoint_type_group': 'category'}
    
    train_df = pd.read_csv(trn_df_path)
    train_lbls = pd.read_csv(trn_lbls_path)
    test_df = pd.read_csv(tst_df_path)
    sub_form = pd.read_csv(sub_form_path)

    lower_features(df_final, test_df)

def simple_clean(df):
    df.index = df['id']
    to_drop = ['id', 'date_recorded', 'scheme_name',
               'extraction_type', 'extraction_type_group',
               'management_group',
               'payment_type',
               'quantity_group',
               'source_type','source_class', 
               'waterpoint_type_group',
               'district_code', 
               'construction_year',
               'num_private',
               'recorded_by']
    df.drop(columns=to_drop, inplace=True)

def add_categories(df)
    for col in ['funder', 'installer', 'subvillage', 'scheme_management']:
        if col=='scheme_management':
            df[col] = df[col].cat.add_categories('Unknown')
        else:
            df[col] = df[col].cat.add_categories('Other')

def fill_nulls(df):
    df.fillna({'funder':'Other',
                'installer': 'Other',
                'subvillage': 'Other', 
                'public_meeting': False,
                'scheme_management': 'Unknown',
                'permit': False}
                , inplace=True)

def encode_me(train_df, test_df, n=20):
    cats = df.select_dtypes(include='category')
    nums = df.select_dtypes(exclude='category')
    lower_features(train_df, test_df)

def lower_features(train_df, test_df):
    for col in ['wpt_name', 'subvillage', 'installer', 'ward', 'funder', 'lga']:
        # get top 20 categories for the column
        temp = train_df[col].value_counts().head(20).keys()
        # if the value is not in the top 20, convert to 'Other'
        train_df[col] = train_df[col].apply(lambda x: 'Other' if x not in temp else x)
        test_df[col] = test_df[col].apply(lambda x: 'Other' if x not in temp else x)


df_final[['public_meeting', 'permit']] = df_final[['public_meeting', 'permit']].astype('boolean')
df_final[['funder', 'installer', 'wpt_name', 'subvillage', 'lga', 'ward']] = df_final[
    ['funder', 'installer', 'wpt_name', 'subvillage', 'lga', 'ward']].astype('category')

test_df[['funder', 'installer', 'wpt_name', 'subvillage', 'lga', 'ward']] = test_df[
    ['funder', 'installer', 'wpt_name', 'subvillage', 'lga', 'ward']].astype('category')

cats = df_final.select_dtypes(include='category')
nums = df_final.select_dtypes(exclude='category')

cats_test = test_df.select_dtypes(include='category')
nums_test = test_df.select_dtypes(exclude='category')

ohe = OneHotEncoder(handle_unknown='ignore')
# Fit categories to training data
ohe.fit(cats)

train_ohe = pd.DataFrame(ohe.transform(cats).toarray(), columns = ohe.get_feature_names(), index=cats.index)
test_ohe = pd.DataFrame(ohe.transform(cats_test).toarray(), columns = ohe.get_feature_names(), index = cats_test.index)

df_final = nums.merge(train_ohe, left_index=True, right_index=True, how='inner')
test_final = nums_test.merge(test_ohe, left_index=True, right_index=True, how='inner')


def save_me(train_df, test_df, output):
    current_time = dt.datetime.now().strftime("%d%m%Y_%I%M%p")

    train_df.to_pickle(output+'training_set_cleaned'+current_time+'.pkl')
    test_df.to_pickle(output+'testing_set_cleaned'+current_time+'.pkl')