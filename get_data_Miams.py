import pandas as pd
import numpy as np

def get_data():

    df = pd.read_csv('Miams_F.csv',sep=";")

    df = df.drop(["Timestamp",'tonic_mean', 'tonic_std', 'tonic_min', 'tonic_max', 'tonic_std','tonic_skewness','tonic_peaks','tonic_rms','tonic_kurtosis'], axis=1)

    df = df.dropna(subset=['hrv_mean_hr',"hrv_min_hr","hrv_max_hr","hrv_std_hr","hrv_rms"])

    df['Stress'] = df['Noon_Morning-Stress'].fillna(df['Evening_Stress'])

    df = df.drop(['Evening_Stress',"Morning_Shape","Morning_Sickness","Morning_Sleep","Noon_Morning-Stress","Evening_Negative_Valency","Evening_Positive_Valency"], axis=1)

    df = df.dropna(subset=['Stress'])


    df= df[df['UUID'] != 86]  
    df= df[df['UUID'] != 87]  
    df= df[df['UUID'] != 107]  
    df= df[df['UUID'] != 116]  

    lenght = df['UUID'].value_counts().sort_index().astype(int).tolist()


    def Norm(list_name_columns, df, length):
        
        for i in list_name_columns:
            if 'UUID'==i:
                continue
            add = 0
            for ind in range(len(lenght)):
                
                end = add + length[ind]

                min_value = df.loc[add:end, i].min()
                df.loc[add:end, i] = df.loc[add:end, i] + abs(min_value)

                df.loc[add:end, i] = (df.loc[add:end, i] - df.loc[add:end, i].min()) / (df.loc[add:end, i].max() - df.loc[add:end, i].min())
                add = end

        return df

    df=Norm(df.columns.tolist(),df,lenght)

    df["Stress"] = np.where(df["Stress"] < 0.33, 0, np.where(df["Stress"] > 0.7, 1, df["Stress"]))

    df = df[(df["Stress"] == 0) | (df["Stress"] == 1)]

    length = df['UUID'].value_counts().sort_index().astype(int).tolist()

    label=df["Stress"].tolist()
    df = df.drop(['UUID',"Stress"],axis=1)

    data=df.values.tolist()

    return data,label,length



def get_data_oversampled():

    df = pd.read_csv('Miams_F.csv',sep=";")

    df = df.drop(["Timestamp",'tonic_mean', 'tonic_std', 'tonic_min', 'tonic_max', 'tonic_std','tonic_skewness','tonic_peaks','tonic_rms','tonic_kurtosis'], axis=1)

    df = df.dropna(subset=['hrv_mean_hr',"hrv_min_hr","hrv_max_hr","hrv_std_hr","hrv_rms"])

    df['Stress'] = df['Noon_Morning-Stress'].fillna(df['Evening_Stress'])

    df = df.drop(['Evening_Stress',"Morning_Shape","Morning_Sickness","Morning_Sleep","Noon_Morning-Stress","Evening_Negative_Valency","Evening_Positive_Valency"], axis=1)

    df = df.dropna(subset=['Stress'])


    df= df[df['UUID'] != 86]  
    df= df[df['UUID'] != 87]  
    df= df[df['UUID'] != 107]  
    df= df[df['UUID'] != 116]  

    lenght = df['UUID'].value_counts().sort_index().astype(int).tolist()


    def Norm(list_name_columns, df, length):
        
        for i in list_name_columns:
            if 'UUID'==i:
                continue
            add = 0
            for ind in range(len(lenght)):
                
                end = add + length[ind]

                min_value = df.loc[add:end, i].min()
                df.loc[add:end, i] = df.loc[add:end, i] + abs(min_value)

                df.loc[add:end, i] = (df.loc[add:end, i] - df.loc[add:end, i].min()) / (df.loc[add:end, i].max() - df.loc[add:end, i].min())
                add = end

        return df

    df=Norm(df.columns.tolist(),df,lenght)

    df["Stress"] = np.where(df["Stress"] < 0.33, 0, np.where(df["Stress"] > 0.7, 1, df["Stress"]))

    df = df[(df["Stress"] == 0) | (df["Stress"] == 1)]

    length = df['UUID'].value_counts().sort_index().astype(int).tolist()

    label=df["Stress"].tolist()
    df = df.drop(['UUID',"Stress"],axis=1)

    data=df.values.tolist()

    drivers=length

    #df:LABEL

    from imblearn.over_sampling import RandomOverSampler

    columns = []
    start_index = 0
    for count in drivers:
        columns.append(label[start_index:start_index + count])
        start_index += count

    df = pd.DataFrame(columns).transpose()

    #df_:data

    columns = []
    start_index = 0
    for count in drivers:
        columns.append(data[start_index:start_index + count])
        start_index += count

    df_ = pd.DataFrame(columns).transpose()

    # Delete students with one label

    valid_columns = df.nunique(dropna=True) > 1

    df = df.loc[:, valid_columns]
    df_ = df_.loc[:, valid_columns]


    oversampled_data_columns = {}
    oversampled_label_columns = {}

    ros = RandomOverSampler(random_state=42)

    for col in df.columns:
        # Extract the label column and corresponding data
        label_col = df[col].dropna()
        data_col = pd.DataFrame(df_[col].dropna().to_list())  # Convert list of lists to DataFrame

        # Oversample the data
        if not label_col.empty:
            data_resampled, labels_resampled = ros.fit_resample(data_col, label_col)
            oversampled_data_columns[col] = pd.DataFrame(data_resampled).values.tolist()
            oversampled_label_columns[col] = pd.Series(labels_resampled).values.tolist()
    

    # Combine resampled data and labels into DataFrames
    oversampled_data_df = pd.DataFrame.from_dict(oversampled_data_columns, orient='index').transpose()
    oversampled_labels_df = pd.DataFrame.from_dict(oversampled_label_columns, orient='index').transpose()

    # Flatten the DataFrames
    label = oversampled_labels_df.values.flatten().tolist()
    label = [x for x in label if pd.notnull(x)]  # Remove NaN values

    data = oversampled_data_df.values.flatten().tolist()
    data = [row for row in data if isinstance(row, list) and all(pd.notnull(element) for element in row)]

    # Recalculate drivers count after oversampling
    drivers = oversampled_labels_df.notna().sum(axis=0).tolist()


    return data,label,drivers



