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
    '''
    Set 'id' as the index and drop specified columns. 
    Columns were dropped based on features importance and 
    correlation.
    '''
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
    '''
    Add 'Unknown' or 'Other' to columns with category dtype.
    This allows for imputing nulls.
    '''
    for col in ['funder', 'installer', 'subvillage', 'scheme_management']:
        if col=='scheme_management':
            df[col] = df[col].cat.add_categories('Unknown')
        else:
            df[col] = df[col].cat.add_categories('Other')

def fill_nulls(df):
    '''
    Impute nulls with the specified fillna dictionary.
    '''
    df.fillna({'funder':'Other',
                'installer': 'Other',
                'subvillage': 'Other', 
                'public_meeting': False,
                'scheme_management': 'Unknown',
                'permit': False}
                , inplace=True)

def encode_me(train_df, test_df):
    '''
    Take categorical columns and one hot encode columns.
    Return concatenation of numerical columns and encoded columns.
    '''
    trn_cats = train_df.select_dtypes(include='category')
    trn_nums = train_df.select_dtypes(exclude='category')
    tst_cats = test_df.select_dtypes(include='category')
    tst_nums = test_df.select_dtypes(exclude='category')

    trn_cats, tst_cats = ohe_me(trn_cats, tst_cats)

    train_df = trn_nums.merge(trn_cats, left_index=True, right_index=True, how='inner')
    test_df = tst_nums.merge(tst_cats, left_index=True, right_index=True, how='inner')

    return train_df, test_df

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

def lower_features(train_df, test_df, n=20):
    '''
    Lower categories in columns to the specified number of categories (default=20).
    For the specified columns, it will take the top n categories and re-categorize
    remaining categories to the specified value.
    This step is done to reduce dimensionality.
    '''
    for col in ['wpt_name', 'subvillage', 'installer', 'ward', 'funder', 'lga']:
        temp = train_df[col].value_counts().head(n).keys()
        train_df[col] = train_df[col].apply(lambda x: 'Other' if x not in temp else x)
        test_df[col] = test_df[col].apply(lambda x: 'Other' if x not in temp else x)
    cols = ['funder', 'installer', 'wpt_name', 'subvillage', 'lga', 'ward']
    train_df[cols] = train_df[cols].astype('category')
    test_df[cols] = test_df[cols].astype('category')

def save_cleaned_data(train_df, test_df):
    '''
    Saves cleaned training and testing set as .pkl files with the current
    date and time. Date and time is in the format of DDMMYY_HHMM, with hours
    and minutes including AM or PM.
    '''
    current_time = dt.datetime.now().strftime("%d%m%Y_%I%M%p")
    train_df.to_pickle(output+'train_set_cleaned'+current_time+'.pkl')
    test_df.to_pickle(output+'test_set_cleaned'+current_time+'.pkl')
    print(f"Saved time: {current_time}")


# Dtype Dictionary used when importing csv files.
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
    '''
    Load and apply cleaning processes to the specified path variables and 
    dtype dictionary. Will save the cleaned dataframes to the specified 
    output in config.ini
    '''
    # import data
    train_df = pd.read_csv(trn_df_path, dtype=dtype_dict)
    train_lbls = pd.read_csv(trn_lbls_path, dtype=dtype_dict)
    test_df = pd.read_csv(tst_df_path, dtype=dtype_dict)
    sub_form = pd.read_csv(sub_form_path, dtype=dtype_dict)
    # remove specified columns + index='id'
    simple_clean(train_df)
    simple_clean(test_df)
    # add categories to specified columns
    add_categories(train_df)
    add_categories(test_df)
    # fill null values with specified dictionary
    fill_nulls(train_df)
    fill_nulls(test_df)
    # set specified columns up to top 25 categories + 'Other'/'Unknown'
    lower_features(train_df, test_df, 25)
    # One hot encode categoricals
    train_df, test_df = encode_me(train_df, test_df)
    # merge training data and labels
    train_df = train_df.merge(train_lbls, left_index=True, right_on='id')
    # save cleaned training and testing sets
    save_cleaned_data(train_df, test_df)
    
get_cleaned_sets(trn_df_path, trn_lbls_path, tst_df_path, sub_form_path, dtype_dict)