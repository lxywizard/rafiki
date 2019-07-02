from typing import Dict, Any

class BudgetOption():
    GPU_COUNT = 'GPU_COUNT'
    TIME_HOURS = 'TIME_HOURS'
    MODEL_TRIAL_COUNT = 'MODEL_TRIAL_COUNT'

Budget = Dict[BudgetOption, Any]
ModelDependencies = Dict[str, str]

class ModelAccessRight():
    PUBLIC = 'PUBLIC'
    PRIVATE = 'PRIVATE'

class InferenceJobStatus():
    STARTED = 'STARTED'
    RUNNING = 'RUNNING'
    ERRORED = 'ERRORED'
    STOPPED = 'STOPPED'

class TrainJobStatus():
    STARTED = 'STARTED'
    RUNNING = 'RUNNING'
    STOPPED = 'STOPPED'
    ERRORED = 'ERRORED'

class TrialStatus():
    PENDING = 'PENDING'
    RUNNING = 'RUNNING'
    ERRORED = 'ERRORED'
    COMPLETED = 'COMPLETED'

class UserType():
    SUPERADMIN = 'SUPERADMIN'
    ADMIN = 'ADMIN'
    MODEL_DEVELOPER = 'MODEL_DEVELOPER'
    APP_DEVELOPER = 'APP_DEVELOPER'

class ServiceType():
    TRAIN = 'TRAIN'
    ADVISOR = 'ADVISOR'
    PREDICT = 'PREDICT'
    INFERENCE = 'INFERENCE'

class ServiceStatus():
    STARTED = 'STARTED'
    DEPLOYING = 'DEPLOYING'
    RUNNING = 'RUNNING'
    ERRORED = 'ERRORED'
    STOPPED = 'STOPPED'
    
class ModelDependency():
    TENSORFLOW = 'tensorflow'
    KERAS = 'Keras'
    SCIKIT_LEARN = 'scikit-learn'
    TORCH = 'torch'
    TORCHVISION = 'torchvision'
    SINGA = 'singa'
    XGBOOST = 'xgboost'
