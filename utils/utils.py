import os
import sys
import torch
import logging
from models import model_dict as MODEL_DICT


def get_config(model_path):
    config_dict = {}
    return config_dict


def load_model(model_path):
    print('==> loading model')
    state_dict = torch.load(model_path)
    model_name, n_cls = state_dict["name"], state_dict["num_head"]

    model = MODEL_DICT[model_name](num_classes=n_cls)
    model.load_state_dict(state_dict['model'])
    print('==> done')
    return model


def init_logging(log_root, models_root):
    log_root.setLevel(logging.INFO)
    formatter = logging.Formatter("Training: %(asctime)s-%(message)s")
    handler_file = logging.FileHandler(os.path.join(models_root, "training.log"))
    handler_stream = logging.StreamHandler(sys.stdout)
    handler_file.setFormatter(formatter)
    handler_stream.setFormatter(formatter)
    log_root.addHandler(handler_file)
    log_root.addHandler(handler_stream)
