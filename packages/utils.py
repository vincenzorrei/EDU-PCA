import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# check if in a column there are only two unique values
def categorical_binary_check(df):
    """
    Checks if a column has only two unique values.
    """
    binary_columns = []
    for column in df.columns:
        if df[column].nunique() == 2 and df[column].dtype in ["object", "category"]:
            binary_columns.append(column)
    return binary_columns


def transform_binary(df, binary_columns):
    """
    Transforms binary columns to 0 and 1.
    """
    for column in binary_columns:
        df[column] = df[column].map(
            {df[column].unique()[0]: 0, df[column].unique()[1]: 1}
        )
    return df


def transform_binary_2(df, binary_columns):
    """
    Transforms binary columns to 0 and 1.
    """
    for column in binary_columns:
        df[column] = df[column].apply(lambda x: 1 if x == df[column].unique()[1] else 0)
    return df


def standard_OHE(df, show=True):
    """
    One-hot encodes the categorical columns in a DataFrame.
    """
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False)
    categorical_binary_columns = categorical_binary_check(df)
    cat_not_binary_columns = [
        col for col in categorical_columns if col not in categorical_binary_columns
    ]

    one_hot_encoded = encoder.fit_transform(df[cat_not_binary_columns])
    if show:
        print(f"Categorical cols:{cat_not_binary_columns}")
        print(f"Binary cols:{categorical_binary_columns}")
    one_hot_df = pd.DataFrame(
        one_hot_encoded, columns=encoder.get_feature_names_out(cat_not_binary_columns)
    )

    # transform the binary columns
    df_binary = transform_binary(df, categorical_binary_columns)

    # merge the two dataframes
    df_encoded = pd.merge(df_binary, one_hot_df, left_index=True, right_index=True)

    # df_encoded = pd.merge([df_binary, one_hot_df], axis=1)
    df_encoded = df_encoded.drop(cat_not_binary_columns, axis=1)
    return df_encoded
