import importlib


def create_obj(cfg, field, addtl_params=None, extra_kwargs=None):
    """
    Creates an object for the data type specified in field param of the config file.

    :param cfg: Config file of the ablation (toml file)
    :param field: field name which contains the info of the object
    :param addtl_params: Any positional "params" override (mainly for the optimizer)
    :param extra_kwargs: Extra constructor kwargs not present under cfg[field]["params"],
                          e.g. theta (which lives under cfg["training"]["params"])
    """
    class_path = cfg[field]["name"]
    cls_params = dict(cfg[field]["params"])  # copy: avoid mutating cfg in place

    module_name, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)

    if addtl_params is not None:
        cls_params["params"] = addtl_params
    if extra_kwargs is not None:
        cls_params.update(extra_kwargs)

    cls = getattr(mod, class_name)
    return cls(**cls_params)
