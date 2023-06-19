from typing import Tuple, Dict

import lightning as L
import torch
import hydra                                            
from omegaconf import DictConfig

                                                                                    
from gold import utils

from pathlib import Path
import os

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:

    #print(cfg.ckpt_path)
    assert cfg.ckpt_fol

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
    }

    if cfg.get("compile"):
        model = torch.compile(model)
                                                                 
    ckpt_fol = cfg.get("ckpt_fol")
    #print(f"-------------------------{ckpt_fol}--------------")
    #print(f'==========={max(Path(ckpt_fol).glob("*.ckpt"), key=os.path.getctime)}============')
    #print(type(cfg))
    if ckpt_fol:
        try:
            ckpt_path = max(Path(ckpt_fol).rglob("*.ckpt"), key=os.path.getctime)
            print(f"Using checkpoint: {ckpt_path}")
        #print(f"------------{ckpt_path}-----------")
        except:
            raise FileNotFoundError("Checkpoint does not exist")


    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    metric_dict = trainer.callback_metrics

    #metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    # evaluate the model
    metric_dict, _ = evaluate(cfg)

    # this will be used by hydra later for optimization
    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
