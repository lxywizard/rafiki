import os
import argparse
from pprint import pprint

from rafiki.client import Client
from rafiki.config import SUPERADMIN_EMAIL
from rafiki.constants import BudgetOption, TaskType, ModelDependency

from examples.scripts.utils import gen_id
from examples.datasets.image_classification.load_cifar_format import load_cifar10

def train_densenet(client, train_dataset_path, val_dataset_path, gpus, hours):    
    '''
        Conducts training of model `PyDenseNetBc` on the CIFAR-10 dataset for the task `IMAGE_CLASSIFICATION`.
        Demonstrates hyperparameter tuning with parameter sharing on Rafiki. 
    '''
    task = TaskType.IMAGE_CLASSIFICATION

    app_id = gen_id()
    app = 'cifar10_densenet_{}'.format(app_id)
    model_name = 'PyDenseNetBc_{}'.format(app_id)

    print('Preprocessing datasets...')
    load_cifar10(train_dataset_path, val_dataset_path)

    print('Creating & uploading datasets onto Rafiki...')
    train_dataset = client.create_dataset('{}_train'.format(app), task, train_dataset_path)
    pprint(train_dataset)
    val_dataset = client.create_dataset('{}_val'.format(app), task, val_dataset_path)
    pprint(val_dataset)

    print('Creating model...')
    model = client.create_model(
        name=model_name,
        task=TaskType.IMAGE_CLASSIFICATION,
        model_file_path='examples/models/image_classification/PyDenseNetBc.py',
        model_class='PyDenseNetBc',
        dependencies={ 
            ModelDependency.TORCH: '1.0.1',
            ModelDependency.TORCHVISION: '0.2.2'
        }
    )
    pprint(model)

    print('Creating train job...')
    budget = { 
        BudgetOption.TIME_HOURS: hours,
        BudgetOption.GPU_COUNT: gpus
    }
    train_job = client.create_train_job(app, task, train_dataset['id'], val_dataset['id'], budget, models=[model['id']])
    pprint(train_job)

    print('Monitor the train job on Rafiki Web Admin')

    # TODO: Evaluate on test dataset?

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--email', type=str, default=SUPERADMIN_EMAIL, help='Email of user')
    parser.add_argument('--password', type=str, default=os.environ.get('SUPERADMIN_PASSWORD'), help='Password of user')
    parser.add_argument('--gpus', type=int, default=0, help='How many GPUs to use')
    parser.add_argument('--hours', type=float, default=2, help='How long the train job should run for (in hours)')
    out_train_dataset_path = 'data/cifar10_for_image_classification_train.zip'
    out_val_dataset_path = 'data/cifar10_for_image_classification_val.zip'
    (args, _) = parser.parse_known_args()

    # Initialize client
    client = Client()
    client.login(email=args.email, password=args.password)

    train_densenet(client, out_train_dataset_path, out_val_dataset_path, args.gpus, args.hours)