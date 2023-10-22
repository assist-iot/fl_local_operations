import base64
import json
import os
import shutil
from typing import List
from zipfile import ZipFile
from logging import log, INFO

import gridfs
import pkg_resources
from prometheus_client import Info
import psutil
import requests as requests
import tenseal as ts
import torch as torch
from flwr.common import Parameters
from pymongo import MongoClient
from application.config import DATABASE_NAME, DB_PORT, REPOSITORY_ADDRESS, HM_SECRET_FILE
from data_transformation.loader import ModelLoader


class BasicModelLoader(ModelLoader):
    temp_dir = "temp"
    config_path = os.path.join("application", "configurations", "model.json")

    def __init__(self, rep_name=REPOSITORY_ADDRESS, db_name=DATABASE_NAME, db_port=DB_PORT):
        self.rep_name = rep_name
        self.db_name = db_name
        self.db_port = db_port

    def check_library(self, model_name, model_version):
        with requests.get(f"{self.rep_name}/model/meta",
                          params={"model_name": model_name,
                                  "model_version": model_version}) \
                as r:
            return r.json()["meta"]["library"]

    def check_loading_path(self, temp):
        '''Checks how nested was the zipped file in order to load it correctly'''
        nested_files = os.listdir(temp)
        log(INFO,
            f'In model directory there are following files {nested_files}')
        if len(nested_files) == 1 and os.path.isdir(os.path.join(temp, nested_files[0])):
            return self.check_loading_path(os.path.join(temp, nested_files[0]))
        else:
            return temp

    def load(self, model_name, model_version):
        client = MongoClient(self.db_name, self.db_port)
        db = client.local
        db_grid = client.repository_grid
        fs = gridfs.GridFS(db_grid)
        if db.models.find_one(
                {"model_name": model_name, "model_version": model_version}):
            result = db.models.find_one({"model_name": model_name, "model_version":
                                         model_version})
            # add model json configuration to then properly use the model
            file = fs.get(result['model_id']).read()
            with open(f'{self.temp_dir}.zip', 'wb') as f:
                shutil.copyfileobj(file, f)
            with ZipFile(f'{self.temp_dir}.zip', 'r') as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall(f'{self.temp_dir}')
        else:
            with requests.get(f"{self.rep_name}/model"
                              f"/{model_name}/{model_version}",
                              stream=True) as r:
                with open(f'{self.temp_dir}.zip', 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            with ZipFile(f'{self.temp_dir}.zip', 'r') as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall(f'{self.temp_dir}')

    def load_format(self):
        '''
        Load the format of the data that will be accepted by the model
        '''
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'rb') as f:
                model_data = json.load(f)
                return model_data

    def cleanup(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(f'{self.temp_dir}')
        if os.path.exists(f'{self.temp_dir}.zip'):
            os.remove(f'{self.temp_dir}.zip')


class HMSerializer:

    @staticmethod
    def write(file, data):
        data = base64.b64encode(data)
        with open(file, 'wb') as f:
            f.write(data)

    @staticmethod
    def read(file):
        with open(file, "rb") as f:
            data = f.read()
            return base64.b64decode(data)


def ts_tensors_to_parameters(tensors: ts.tensors.ckkstensor.CKKSTensor) -> Parameters:
    """Convert Tenseal tensors to parameters object."""
    tensors = [tensor.serialize() for tensor in tensors]
    return Parameters(tensors=tensors, tensor_type="homomorphic")


def parameters_to_ts_tensors(parameters: Parameters) -> List[ts.tensors.ckkstensor.CKKSTensor]:
    """Convert parameters object to Tenseal tensors."""
    context = ts.context_from(
        HMSerializer.read(
            os.path.join(os.sep, "code", "application", "src", "custom_clients", "hm_keys", HM_SECRET_FILE)))
    return [ts.ckks_tensor_from(context, tensor) for tensor in parameters.tensors]


def check_storage():
    """Checks how much storage is left on the device"""
    _, _, free = shutil.disk_usage("/")
    value = free // (2 ** 30)
    return value


def check_memory():
    """Checks how much memory is left on the device"""
    memory = psutil.virtual_memory()
    free = memory.free
    value = free // 1000000000
    return value


def check_gpu():
    """Checks if cuda available and assumes it's not if torch is also not there"""
    value = False
    try:
        import torch
    except ImportError:
        return value
    else:
        value = torch.cuda.is_available()
        return value


def check_packages():
    """"Checks out the dict of installed packages along with versions"""
    package_names = [d.project_name for d in pkg_resources.working_set]
    return {key: pkg_resources.get_distribution(key).version for key in package_names}


def check_models():
    """When we decide to store specific kinds of data in the local database
    We'll define their retrieval here"""
    return {}
