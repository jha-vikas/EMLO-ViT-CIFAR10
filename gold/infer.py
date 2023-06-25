from typing import Tuple, Dict

import lightning as L
import torch
import hydra                                            
from omegaconf import DictConfig
from torchvision import transforms as T
from PIL import Image
import torch.nn.functional as F
                                                                                    
from gold import utils

from pathlib import Path
import os

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def infer(cfg: DictConfig) -> Tuple[dict, dict]:

    #print(cfg.ckpt_path)
    assert cfg.ckpt_fol
    assert cfg.img_path #image for inference

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    object_dict = {
        "cfg": cfg,
        "model": model,
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

    ckpt = torch.load(ckpt_path)

    log.info(f"Loading model from checkpoint {ckpt_path}")
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    log.info(f"Checkpoint loaded")


    categories = [
        "cat",
        "dog",
    ]
    
    transforms = T.Compose(
        [
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(cfg.img_path)
    img = transforms(img).unsqueeze(0)

    logits = model(img)
    preds = F.softmax(logits, dim=1).squeeze(0).tolist()

    pred_dict = {i:j for i,j in zip(categories, preds)}

    if len(pred_dict) > 5:
        pred_dict = dict(sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)[:5])

    print("\n")
    print(f"Predicted probability: {pred_dict}")
    print("\n")

    return pred_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig):
    # evaluate the model
    infer(cfg)


if __name__ == "__main__":
    main()
