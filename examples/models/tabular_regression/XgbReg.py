import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pickle
import base64
import numpy as np

import sys
sys.path.insert(0, '/Users/pro/Desktop/rafiki')

from rafiki.model import BaseModel, IntegerKnob, FloatKnob, utils
from rafiki.constants import ModelDependency
from rafiki.model.dev import test_model_class

class XgbReg(BaseModel):
    '''
    Implements a decision tree classifier on Scikit-Learn for image classification
    '''
    @staticmethod
    def get_knob_config():
        return {
            'n_estimators': IntegerKnob(0, 100),
            'min_child_weight': IntegerKnob(1, 6),
            'max_depth': IntegerKnob(1, 10),
            'gamma': FloatKnob(0.0, 1.0, is_exp=False),
            'subsample': FloatKnob(0.5, 1.0, is_exp=False),
            'colsample_bytree': FloatKnob(0.1, 0.7, is_exp=False)
        }

    def __init__(self, **knobs):
        self.__dict__.update(knobs)
        self._clf = self._build_classifier(self.n_estimators, self.min_child_weight, \
            self.max_depth, self.gamma, self.subsample, self.colsample_bytree)
       
    def train(self, dataset_path, **kwargs):
        dataset = utils.dataset.load_dataset_of_tabular_format(dataset_path)
        data = dataset.data
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]
        self._clf.fit(X, y)

        # Compute train accuracy
        preds = self._clf.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        utils.logger.log('Train RMSE: {}'.format(rmse))

    def evaluate(self, dataset_path):
        dataset = utils.dataset.load_dataset_of_tabular_format(dataset_path)
        data = dataset.data
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]
        preds = self._clf.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        return rmse

    def predict(self, queries):
        pass

    def dump_parameters(self):
        params = {}

        # Put model parameters
        clf_bytes = pickle.dumps(self._clf)
        clf_base64 = base64.b64encode(clf_bytes).decode('utf-8')
        params['clf_base64'] = clf_base64

        return params

    def load_parameters(self, params):
        # Load model parameters
        clf_base64 = params['clf_base64']
        clf_bytes = base64.b64decode(clf_base64.encode('utf-8'))
        self._clf = pickle.loads(clf_bytes)

    def _build_classifier(self, n_estimators, min_child_weight, max_depth, gamma, subsample, colsample_bytree):
        clf = xgb.XGBRegressor(
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            max_depth=max_depth,
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree
        ) 
        return clf

if __name__ == '__main__':
    test_model_class(
        model_file_path=__file__,
        model_class='XgbReg',
        task='TABULAR_REGRESSION',
        dependencies={
            ModelDependency.XGBOOST: '0.90'
        },
        train_dataset_path='data/boston_train.csv',
        val_dataset_path='data/boston_val.csv',
        test_dataset_path='data/boston_test.csv',
        budget={ 'MODEL_TRIAL_COUNT': 10},
        task_type='regression'
    )
