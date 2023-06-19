# timm model based CIFAR10 trainer

### To run the model
Steps:
1. Install the package using:
```
pip install .
```
To install in developemtn mode:
```
pip install -e .
```

2. To run training.
```
gold_train
```
3. To run evaluation:
```
gold_eval ckpt_fol=<location of checkpoint folder>
```
    Note: The evaluation recursively searches for checkpoint file in the folder and subfolders provided above, and runs evaluation using the latest file. In case one wants to search in all possible folders, provde the ckpt_fol=".", else provide the folder where the checkpoint file one wants to use is localted.

    Example:
    ```
    gold_eval ckpt_fol=./outputs
    ```

4. The parameters in train.yaml & eval.yaml files in configs folder can be overriden.
    Example:
    ```
    gold_train <parameter in train.yaml> = <new parameter value>
    ``` 

### Docker

To create the docker file, run:
```
docker build --tag gold:latest .
```
To run docker image:
```
docker run gold:latest
