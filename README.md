# ViT based image classification model

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

5. To run inference:

```
gold_infer ckpt_fol=<location of checkpoint folder> img_path=<image file location>
```


6. To add data to dvc:
    - In case dvc is not instantiated, run:
    ```
    dvc init
    ```
    - Add remote to dvc
    ```
    dvc remote add --default <remote name> <remote location>
    ```
    Eg:- If local folder has to be added as remote, naming remote as 'local':
    ```
    dvc remote add -d local <location of local folder for dvc>
    ```
    Note: add '--default' to make it default dvc remote location. Other than local folder, google drive, amazon s3, MS-azuer blob storage, ssh, GCP, HDFS, & HTTP can be used as remote for dvc.
    - Add data files to be tracked by dvc:
    ```
    dvc add data
    dvc add outputs
    ```
    - Add dvc to gitignore:
    ```
    git add data.dvc .gitignore
    ```
    - Push dvc to the remote location:
    ```
    dvc pull -r <dvc remote name>
    ```
    - To move to specific file revision:
    ```
    git checkout <...>
    dvc checkout
    ```

### Docker

To create the docker file, run:
```
docker build --tag gold:latest .
```
To run docker image:
```
docker run gold:latest
