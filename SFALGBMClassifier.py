from SFAClassifier import SFAClassifier
from lightgbm import Dataset, train, Booster
from sklearn.metrics import roc_auc_score as auc
import warnings
import numpy as np


class SFALGBMClassifier(SFAClassifier):

    def __init__(self, ds_name, seed):
        super().__init__(ds_name, seed)  # Initialize the parent class
        self.model_name = 'lgbm'  # Set the model name to 'lgbm'

    def objective(self, trial):
        """
        Hyperparameters optimization
        :param trial: the current trial
        :return: the auc score achieved in the trial
        """
        train_x, train_y = self.get_train_data()  # Get training data
        valid_x, valid_y = self.get_test_data()  # Get validation data
        dtrain = Dataset(train_x, label=train_y, categorical_feature=self.categories) if self.categories is not None else Dataset(train_x, label=train_y)  # Create LightGBM dataset
        valid_y_np = self.get_y_np(valid_y)  # Convert validation labels to numpy array

        sub_samples_l, sub_samples_h = self.get_high_low_subsamples()  # Get subsample range
        col_sample_bytree_l, col_sample_bytree_h = self.get_high_low_col_samples()  # Get column sample range

        params = {
            "device_type": 'cuda',  # Use GPU for training"
            'objective': self.get_task(),  # Set the task type
            'verbosity': -1,  # Suppress LightGBM output
            'min_gain_to_split': 0.0001,  # Minimum gain to split
            'num_classes': self.get_num_classes(),  # Number of classes
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.0001, 0.03),  # Learning rate
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-4, 40.0, log=True),  # L2 regularization
            'max_depth': trial.suggest_int('max_depth', 3, 11),  # Maximum depth of trees
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', sub_samples_l, sub_samples_h),  # Bagging fraction
            'feature_fraction': trial.suggest_uniform('feature_fraction', col_sample_bytree_l, col_sample_bytree_h),  # Feature fraction
        }
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')  # Ignore warnings
            bst = train(params, dtrain, num_boost_round=int((10 / (0.01 + params["learning_rate"]) ** 2) / 5))  # Train the model
        probas = bst.predict(valid_x)  # Predict probabilities
        auc_score = auc(valid_y_np, probas, multi_class='ovo') if self.get_n_classes() > 2 else auc(valid_y_np, probas)  # Calculate AUC score
        return auc_score  # Return AUC score

    def train(self, x_train, y_train):
        """
        Initialize LGBM classifier and train it
        :param x_train: train features
        :param y_train: train target
        :return: the trained classifier
        """
        params = self.get_hyper_params()  # Get hyperparameters
        dtrain = Dataset(x_train, label=y_train, categorical_feature=self.categories) if self.categories is not None else Dataset(x_train, label=y_train)  # Create LightGBM dataset
        params['verbosity'] = -1  # Suppress LightGBM output
        params['num_classes'] = self.get_num_classes()  # Number of classes
        params['objective'] = self.get_task()  # Set the task type
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')  # Ignore warnings
            model = train(params, dtrain, num_boost_round=int((10 / (0.01 + params["learning_rate"]) ** 2) / 5))  # Train the model
        return model  # Return the trained model

    def predict_proba(self, clf, val_data):
        """
        Return the predicted probability for the given classifier.
        :param clf: LGBM classifier
        :param val_data: data
        :return: val_data's predicted probability
        """
        x_val = val_data[0]  # Get validation features
        probs = clf.predict(x_val)  # Predict probabilities
        if self.get_n_classes() == 2:
            probs = np.array([np.array([1 - i, i]) for i in probs])  # Adjust probabilities for binary classification
        return probs  # Return probabilities

    def get_task(self):
        """
        Return the task based on the amount of classed in the data
        :return: binary if there are two classed and 'multiclass' otherwise
        """
        return 'binary' if self.get_n_classes() == 2 else 'multiclass'  # Return task type

    def save_model(self, clf, path):
        """
        Saved the model in .model format
        :param clf: LGBM classifier
        :param path: path to save the model in
        """
        clf.save_model(path+'.model')  # Save the model

    def get_num_classes(self):
        """Return the number of classes"""
        return 1 if self.get_n_classes() == 2 else self.get_n_classes()  # Return number of classes

    def load_model(self, path):
        """
        Load the LGBM classifier from the given path
        :param path: path
        :return: LGBM classifier
        """
        booster = Booster(model_file=path + '.model')  # Load the model
        booster.params['objective'] = self.get_task()  # Set the task type
        return booster  # Return the loaded model
