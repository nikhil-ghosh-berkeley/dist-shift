import hydra
from omegaconf import DictConfig

from src.train import train
from src.utils import (
    filter_config,
    get_dict_hash,
    get_class_name,
    Categorizer,
    load_pickle,
    dump_pickle,
)
from pytorch_lightning import seed_everything
import os


@hydra.main(config_path="conf/", config_name="config.yaml")
def main(config: DictConfig):
    if config.eval_last_epoch_only:
        val_check_interval = config.trainer.val_check_interval
        assert isinstance(val_check_interval, float) and (val_check_interval == 1.0)

    # extract data and model experiment info to group runs
    group_dict = dict(filter_config(config.datamodule), **filter_config(config.model))
    group_dict["name"] = get_class_name(config.datamodule._target_, "train")
    group_hash = get_dict_hash(group_dict)
    config.logger.group = group_hash

    if not os.path.isdir(config.pred_save_path):
        os.mkdir(config.pred_save_path)

    hash_dict_fname = os.path.join(config.pred_save_path, "hash_dict.pkl")
    if os.path.isfile(hash_dict_fname):
        hash_dict = load_pickle(hash_dict_fname)
    else:
        hash_dict = dict()

    hash_dict[group_hash] = group_dict
    dump_pickle(hash_dict, hash_dict_fname)

    cats_fname = os.path.join(config.pred_save_path, "cats.pkl")
    if os.path.isfile(cats_fname):
        cats = load_pickle(cats_fname)
    else:
        cats = {
            "seed": Categorizer(),
            "hash": Categorizer(),
            "name": Categorizer(),
        }

    cats["hash"].add(group_hash)

    for val_name in config.val_names:
        cats["name"].add(val_name)

    config.seed = seed_everything()
    cats["seed"].add(config.seed)

    dump_pickle(cats, cats_fname)

    # start training
    train(config)


if __name__ == "__main__":
    main()