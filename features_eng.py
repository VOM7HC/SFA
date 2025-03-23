import featuretools as ft
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA

"""Other feature augmentation methods"""

def features_tools_extend_X(x, y, dataset_name):
    """
    Calculate augmented features using FeatureTools and use feature selection to select 2n+1 best features
    :param x: original features
    :param y: original target column
    :param dataset_name: dataset name
    :return: 2n+1 X N matrix of augmented features
    """
    copy_x = x.copy(deep=True)  # Create a deep copy of the original features
    es = ft.EntitySet(id=dataset_name)  # Initialize an EntitySet with the dataset name
    
    # Add the dataframe to the EntitySet
    es.add_dataframe(dataframe_name='data', dataframe=copy_x, index='index')
    
    # Run deep feature synthesis with transformation primitives
    feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='data',
                                          trans_primitives=['multiply_numeric', 'add_numeric', 'modulo_numeric',
                                                            'percentile'])
    
    feature_matrix.fillna(0, inplace=True)  # Fill NaN values with 0
    
    # Select 2n+1 best features using SelectKBest
    x_selectKBest = SelectKBest(mutual_info_classif, k=x.shape[1]*2+1).fit(feature_matrix, y.to_numpy().reshape(-1))
    mask = x_selectKBest.get_support()  # Get the mask of selected features
    selected_features = list(feature_matrix.columns[mask])  # List of selected feature names
    x_new = feature_matrix[selected_features]  # Create new feature matrix with selected features
    
    return x_new  # Return the new feature matrix

def pca_extend_X(x):
    """
    Calculate PCA features and add them to the original features
    :param x: original features
    :return: 2n X N matrix of augmented features
    """
    pca = PCA()  # Initialize PCA
    principalComponents = pca.fit_transform(x)  # Fit and transform the original features
    principalDf = pd.DataFrame(data=principalComponents, columns=[col + '_pca' for col in x])  # Create a DataFrame with PCA features
    new_x = x.join(principalDf)  # Join the original features with PCA features
    
    return new_x  # Return the new feature matrix
