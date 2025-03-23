from SFAClassifier import SFAClassifier
from xgboost import DMatrix, train, Booster
import numpy as np


class SFARandomForestClassifier(SFAClassifier):

    def __init__(self, ds_name, seed):
        super().__init__(ds_name, seed)
        self.model_name = 'random_forest'  # Set the model name to 'random_forest'

    def objective(self, trial):
        pass  # Placeholder for the objective function

    def get_hyper_params(self):
        """
        Return the hyperparameters of random forest using the XGBoost implementation
        """
        params = {
                  'verbosity': 0,  # Set verbosity level to 0
                  'objective': self.get_task(),  # Set the objective based on the task
                  'num_class': 1 if self.get_n_classes() == 2 else self.get_n_classes(),  # Set number of classes
                  'colsample_bynode': np.sqrt(self.get_n_features()) / self.get_n_features(),  # Set column sample by node
                  'learning_rate': 1,  # Set learning rate
                  'num_parallel_tree': 250,  # Set number of parallel trees
                  'subsample': 0.63,  # Set subsample ratio
                  'tree_method': 'gpu_hist'  # Use GPU histogram tree method
        }
        return params

    def train(self, x_train, y_train):
        """
        Initialize random forest classifier and train it
        :param x_train: train features
        :param y_train: train target
        :return: the trained classifier
        """
        params = self.get_hyper_params()  # Get hyperparameters
        params['num_class'] = 1 if self.get_n_classes() == 2 else self.get_n_classes()  # Set number of classes again
        dtrain = DMatrix(x_train, label=y_train, enable_categorical=True) if self.categories is not None else DMatrix(x_train, label=y_train)  # Create DMatrix for training data
        return train(params=params, dtrain=dtrain, num_boost_round=1)  # Train the model

    def predict_proba(self, clf, val_data):
        """
        Return the predicted probability for the given classifier.
        :param clf: LGBM classifier
        :param val_data: data
        :return: val_data's predicted probability
        """
        x_val, y_val = val_data[0], val_data[1]  # Extract validation features and target
        dvalid = DMatrix(x_val, label=y_val, enable_categorical=True) if self.categories is not None else DMatrix(x_val, label=y_val)  # Create DMatrix for validation data
        probs = clf.predict(dvalid)  # Predict probabilities
        if self.get_n_classes() == 2:
            probs = np.array([np.array([1 - i, i]) for i in probs])  # Adjust probabilities for binary classification
        return probs

    def get_task(self):
        """
        Return the task based on the amount of classes in the data
        :return: 'binary:logistic' if there are two classes and 'multi:softprob' otherwise
        """
        return 'binary:logistic' if self.get_n_classes() == 2 else 'multi:softprob'  # Determine task type

    @staticmethod
    def save_model(clf, path):
        """
       Save the model in .model format
       :param clf: random forest classifier
       :param path: path to save the model in
       """
        clf.save_model(path + '.model')  # Save the model to the specified path

    @staticmethod
    def load_model(path):
        """
        Load the random forest classifier from the given path
        :param path: path
        :return: random forest classifier
         """
        bst = Booster()  # Create a new Booster instance
        bst.load_model(path + '.model')  # Load the model from the specified path
        return bst  # Return the loaded model

    def get_DMatrix(self, X, y):
        """
        Wrap the dataframe in a DMatrix
        :param X: features
        :param y: target
        :return: data in DMatrix format
        """
        if self.get_categories() is not None:
            return DMatrix(X, label=y, enable_categorical=True)  # Create DMatrix with categorical support
        else:
            return DMatrix(X, label=y)  # Create DMatrix without categorical support
