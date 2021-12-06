import hydra
import os
from omegaconf import DictConfig
from src.train import train
from src.utils import (
    filter_config,
    get_dict_hash,
)
from src.simple_utils import load_pickle, dump_pickle
from os.path import join

@hydra.main(config_path="conf/", config_name="config.yaml")
def main(config: DictConfig):
    if config.eval_last_epoch_only:
        val_check_interval = config.trainer.val_check_interval
        assert isinstance(val_check_interval, float) and (val_check_interval == 1.0)

    # extract data and model experiment info to group runs
    group_dict = dict(filter_config(config.datamodule), **filter_config(config.model))
    # group_dict["name"] = get_class_name(config.datamodule._target_, "train")
    group_hash = get_dict_hash(group_dict)
    config.logger.group = group_hash
    print(group_dict)

    if not os.path.isdir(config.pred_save_path):
        os.mkdir(config.pred_save_path)

    hash_dict_fname = join(config.pred_save_path, "hash_dict.pkl")
    if os.path.isfile(hash_dict_fname):
        hash_dict = load_pickle(hash_dict_fname)
    else:
        hash_dict = dict()

    hash_dict[group_hash] = group_dict
    dump_pickle(hash_dict, hash_dict_fname)

    raw_path = join(config.pred_save_path, "raw")
    if not os.path.isdir(raw_path):
        os.mkdir(raw_path)

    # start training
    train(config)


if __name__ == "__main__":
    main()
