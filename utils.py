import os
import importlib
from recbole.utils import get_model as recbole_get_model


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_model(model_name):
    if importlib.util.find_spec(f"model.{model_name.lower()}", __name__):
        model_module = importlib.import_module(f"model.{model_name.lower()}", __name__)
        model_class = getattr(model_module, model_name)
        return model_class
    else:
        return recbole_get_model(model_name)
