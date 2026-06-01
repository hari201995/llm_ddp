import importlib


def create_obj(cfg, field, addtl_params=None):
    """
    Creates an object for the data type specified in field param of the config file.

    :param cfg: Config file of the ablation (toml file)
    :param field: field name which contains the info of the object
    :param addtl_params: Any kwargs that need to be given for __init__
    """
    class_path = cfg[field]["name"]
    cls_params = cfg[field]["params"]

    # Split path into module and class
    module_name, class_name = class_path.rsplit(".", 1)

    # Dynamically import module
    mod = importlib.import_module(module_name)

    # Add config params
    if addtl_params != None:
        # this is mainly for handling optimizer
        cls_params["params"] = addtl_params

    # Retrieve class object from module
    cls = getattr(mod, class_name)
    obj = cls(**cls_params)
    return obj
