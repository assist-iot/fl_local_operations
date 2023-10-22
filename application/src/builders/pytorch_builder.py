import gc
import logging
import os
from collections import OrderedDict
from logging import log, INFO

import flwr as fl
import numpy as np
import requests as requests
import torch as torch
import torchvision

from application.additional.exceptions import BadConfigurationError
from application.additional.object_loaders import TrainingSetupLoader
from application.additional.utils import BasicModelLoader
from application.config import ORCHESTRATOR_SVR_ADDRESS
from application.src.clientbuilder import FlowerClientInferenceBuilder, FlowerClientTrainingBuilder
from application.src.data_loader import BinaryDataLoader

pytorch_optimizers = {
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sparseadam": torch.optim.SparseAdam,
    "adamax": torch.optim.Adamax,
    "asgd": torch.optim.ASGD,
    "lbfgs": torch.optim.LBFGS,
    "nadam": torch.optim.NAdam,
    "radam": torch.optim.RAdam,
    "rmsprop": torch.optim.RMSprop,
    "rprop": torch.optim.Rprop,
    "sgd": torch.optim.SGD
}
pytorch_schedulers = {
    "lambdalr": torch.optim.lr_scheduler.LambdaLR,
    "multiplicativelr": torch.optim.lr_scheduler.MultiplicativeLR,
    "steplr": torch.optim.lr_scheduler.StepLR,
    "multisteplr": torch.optim.lr_scheduler.MultiStepLR,
    "constantlr": torch.optim.lr_scheduler.ConstantLR,
    "linearlr": torch.optim.lr_scheduler.LinearLR,
    "exponentiallr": torch.optim.lr_scheduler.ExponentialLR,
    "cosineannealinglr": torch.optim.lr_scheduler.CosineAnnealingLR,
    "chainedscheduler": torch.optim.lr_scheduler.ChainedScheduler,
    "sequentiallr": torch.optim.lr_scheduler.SequentialLR,
    "reducelronplateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "cycliclr": torch.optim.lr_scheduler.CyclicLR,
    "onecyclelr": torch.optim.lr_scheduler.OneCycleLR,
    "cosineannealingwarmrestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
}


def collate_fn(batch):
    return tuple(zip(*batch))


class PytorchClient(fl.client.NumPyClient):

    def __init__(self, training_id, configuration, data_loader):
        logger = logging.getLogger()
        # log all messages, debug and up
        logger.setLevel(logging.INFO)
        log(INFO, "Pytorch client created")
        self.configuration = configuration
        self.data_loader = data_loader
        (self.x_train, self.y_train) = self.data_loader.load_train()
        (self.x_test, self.y_test) = self.data_loader.load_test()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.round = 1
        self.training_id = training_id

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        logger = logging.getLogger()
        # log all messages, debug and up
        logger.setLevel(logging.INFO)
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.train()
        for ep in range(self.configuration.config[0].epochs):
            try:
                requests.get(
                    f"{ORCHESTRATOR_SVR_ADDRESS}/recoverTrainingEpochs"f"/{str(ep)}"f"/{str(self.configuration.config[0].epochs)}")
            except requests.exceptions.ConnectionError as e:
                log(INFO,
                    f'Could not connect to orchestrator on {ORCHESTRATOR_SVR_ADDRESS}')
            train_loss = 0
            if self.warmup:
                log(INFO, "Warmup scheduler in action")
                lr_scheduler = self.warmup_scheduler
                self.warmup -= 1
            else:
                lr_scheduler = self.lr_scheduler
            for images, targets in zip(self.x_train, self.y_train):
                images = list(image.to(self.device) for image in [images])
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in [
                    targets]]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict[0].values())

                loss_value = losses.item()
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
                loss_batch = np.round(loss_value, 4)
                train_loss += loss_batch
                gc.collect()
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            log(INFO, 'Loss: ' + str(train_loss))
            lr_scheduler.step()
        return self.get_parameters(config={}), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.eval()
        log(INFO, "Evaluation starting")
        dir_results = os.path.join(os.sep, 'test')
        if not os.path.exists(dir_results):
            os.makedirs(dir_results)
        log(INFO, f"Images produced")
        self.round += 1
        return float(0), len(self.x_train), {}


class PytorchBuilder(FlowerClientTrainingBuilder, FlowerClientInferenceBuilder):

    def __init__(self, id, configuration):
        self.configuration = configuration
        self.library = "pytorch"
        self.id = id

    def prepare_training(self):
        setup_loader = TrainingSetupLoader()
        setup = setup_loader.load_setup()
        loader_id = setup["data_loader"]
        if self.configuration.client_type_id == "base":
            data_loader = BinaryDataLoader()
            client_id = setup["client_library"][self.library]["id"]
            self.client = PytorchClient(self.id, self.configuration, data_loader)
        else:
            data_loader = setup_loader.load_data_loader(loader_id)()
            # Load the right client
            client_id = setup["client_library"][self.library]["id"]
            self.client = setup_loader.load_client(client_id)(
                self.id, self.configuration, data_loader)
        self.client.model = self.add_model()
        self.client.optimizer = self.add_optimizer()
        self.client.lr_scheduler = self.add_scheduler()
        self.client.warmup_scheduler, self.client.warmup = self.add_warmup()
        return self.client

    def prepare_inference(self) -> None:
        return self.add_model()

    def add_model(self):
        loader = BasicModelLoader()
        log(INFO, "Model in loading")
        loader.load(self.configuration.model_name,
                    self.configuration.model_version)
        model = torch.jit.load(
            f'{BasicModelLoader.temp_dir}/scripted_model.pt')
        loader.cleanup()
        log(INFO, "Model loaded")
        return model

    def add_optimizer(self) -> None:
        config = self.configuration.optimizer_config
        input_conf = config.dict(exclude_unset=True)
        input_conf.pop("optimizer")
        params = [p for p in self.client.model.parameters() if p.requires_grad]
        try:
            optimizer = pytorch_optimizers[self.configuration.optimizer_config.optimizer](
                params=params, **input_conf)
        except AttributeError as e:
            raise BadConfigurationError("optimizer")
        log(INFO, "Optimizer added")
        return optimizer

    def add_warmup(self, warmup=False) -> None:
        if warmup:
            config = self.configuration.warmup_config
            input_conf = config.scheduler_conf.dict(exclude_unset=True)
            input_conf.pop("scheduler")

            def warmup_function(x):
                if x >= config.warmup_iters:
                    return 1
                alpha = float(x) / config.warmup_iters
                return config.warmup_factor * (1 - alpha) + alpha

            try:
                warmup_scheduler = pytorch_schedulers[
                    config.scheduler_conf.scheduler](optimizer=self.client.optimizer,
                                                     **input_conf)
            except AttributeError as e:
                raise BadConfigurationError("warmup scheduler")
            warmup = self.configuration.warmup_config.warmup_epochs
        else:
            warmup = 0
            warmup_scheduler = None
        log(INFO, "Warmup added")
        return warmup_scheduler, warmup

    def add_scheduler(self):
        scheduler_conf = self.configuration.scheduler_config
        input_conf = scheduler_conf.dict(exclude_unset=True)
        input_conf.pop("scheduler")
        try:
            lr_scheduler = pytorch_schedulers[
                scheduler_conf.scheduler](optimizer=self.client.optimizer, **input_conf)
        except AttributeError as e:
            raise BadConfigurationError("scheduler")
        log(INFO, "Scheduler added")
        return lr_scheduler
