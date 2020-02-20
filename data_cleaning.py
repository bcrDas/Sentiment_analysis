import pandas as pd



def xlsx_to_csv(input_file,op_file):
        df = pd.read_excel(input_file)
        df1 = df.to_csv(op_file)
        return df1

def data_load_excel(the_file):
    train = pd.read_excel(the_file)
    #print(train.columns.tolist())
    #print(train)
    train_original=train.copy()
    #display(train.head(10))
    return train,train_original

def data_load_csv(the_file):
    #train = pd.read_csv(the_file,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')
    train = pd.read_csv(the_file)
    #print(train.columns.tolist())
    #print(train)
    train_original=train.copy()
    #display(train.head(10))
    return train,train_original

def seeing_column_name(train):
    # iterating the columns 
    for col in train.columns: 
        print(col) 
    print("\n")

def seeing_empty_value(train):
    # Count the Null Columns
    null_columns=train.columns[train.isnull().any()]
    print("Initial Null columns: ",train[null_columns].isnull().sum())
    print("\n")
    return train


def removing_column(train):
    #Removing the unused or irrelevant columns
    to_drop = ['name','retweet_count','tweet_id','tweet_location','user_timezone','tweet_coord']
    train.drop(to_drop,inplace=True,axis=1)
    print(train.columns)
    print("\n")
    return train

def clean_nothing_dataset(train):
    '''
    assert isinstance(train, pd.DataFrame), "train needs to be a pd.DataFrame"
    train.dropna(inplace=True)
    indices_to_keep = ~train.isin([np.nan]).any(1)
    return train[indices_to_keep].astype(np.float64)
    '''
    ##train = train.replace('', np.nan, inplace=True)
    print(train.isnull().sum())
    print("\n")
    train = train.dropna()
    #train = train.fillna(0)
    print(train.isnull().sum())
    print("\n")
    return train

def save_modified_csv(train,op_file):
    # Saving modified dataframe.
    train.to_csv(op_file)
    return train

def save_modified_excel(train,op_file):
    # Saving modified dataframe.
    train.to_excel(op_file)
    return train

if __name__ == '__main__':

    #train_0,train_original_0 = data_load_csv("Tweets.csv")
    train_0,train_original_0 = data_load_excel("Tweets.xlsx")

    seeing_column_name(train_0)
    
    train_1 = removing_column(train_0)

    seeing_empty_value(train_1)

    #train_2 = renaming_column(train_1)

    #train_3 = clean_nothing_dataset(train_1)

    train_2 = save_modified_excel(train_1,'temp.xlsx')
    # Deleteing the column from xlxs file manually(if needed)

    #save_modified_csv(train_1,'temp.csv')
    
