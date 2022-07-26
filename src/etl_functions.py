import configparser
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import datetime as dt

config = configparser.ConfigParser()
config.read("../src/config.ini")

output = config['paths']['data_path']

trn_lbls_path = config['paths']['train_labels']
trn_df_path = config['paths']['train_data']
tst_df_path = config['paths']['test_data']
sub_form_path = config['paths']['sub_form']

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

def add_categories(df):
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

def encode_me(train_df, test_df):
    trn_cats = train_df.select_dtypes(include='category')
    trn_nums = train_df.select_dtypes(exclude='category')
    tst_cats = test_df.select_dtypes(include='category')
    tst_nums = test_df.select_dtypes(exclude='category')

    trn_cats, tst_cats = ohe_me(trn_cats, tst_cats)

    train_df = trn_nums.merge(trn_cats, left_index=True, right_index=True, how='inner')
    test_df = tst_nums.merge(tst_cats, left_index=True, right_index=True, how='inner')

    return train_df, test_df

def ohe_me(trn_cats, tst_cats):
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(trn_cats)
    trn_ohe = pd.DataFrame(ohe.transform(trn_cats).toarray(), 
                            columns = ohe.get_feature_names(), index=trn_cats.index)
    tst_ohe = pd.DataFrame(ohe.transform(tst_cats).toarray(), 
                            columns = ohe.get_feature_names(), index = tst_cats.index)
    return trn_ohe, tst_ohe

def lower_features(train_df, test_df, n):
    for col in ['wpt_name', 'subvillage', 'installer', 'ward', 'funder', 'lga']:
        temp = train_df[col].value_counts().head(n).keys()
        train_df[col] = train_df[col].apply(lambda x: 'Other' if x not in temp else x)
        test_df[col] = test_df[col].apply(lambda x: 'Other' if x not in temp else x)
    cols = ['funder', 'installer', 'wpt_name', 'subvillage', 'lga', 'ward']
    train_df[cols] = train_df[cols].astype('category')
    test_df[cols] = test_df[cols].astype('category')

def save_cleaned_data(train_df, test_df):
    current_time = dt.datetime.now().strftime("%d%m%Y_%I%M%p")
    train_df.to_pickle(output+'train_set_cleaned'+current_time+'.pkl')
    test_df.to_pickle(output+'test_set_cleaned'+current_time+'.pkl')
    print(f"Saved time: {current_time}")

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

def get_cleaned_sets(trn_df_path, trn_lbls_path, tst_df_path, sub_form_path, dtype_dict):
    train_df = pd.read_csv(trn_df_path, dtype=dtype_dict)
    train_lbls = pd.read_csv(trn_lbls_path, dtype=dtype_dict)
    test_df = pd.read_csv(tst_df_path, dtype=dtype_dict)
    sub_form = pd.read_csv(sub_form_path, dtype=dtype_dict)
    simple_clean(train_df)
    simple_clean(test_df)
    add_categories(train_df)
    add_categories(test_df)
    fill_nulls(train_df)
    fill_nulls(test_df)
    lower_features(train_df, test_df, 25)
    train_df, test_df = encode_me(train_df, test_df)
    train_df = train_df.merge(train_lbls, left_index=True, right_on='id')
    save_cleaned_data(train_df, test_df)
    
get_cleaned_sets(trn_df_path, trn_lbls_path, tst_df_path, sub_form_path, dtype_dict)