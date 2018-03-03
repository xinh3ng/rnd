## Smule Challenge
git repo (private): git@github.com:xinh3ng/smule.git

### Set up environment
* $ virtualenv venv -p python3
* $ source venv/bin/activate (Activate virtual env)
* $ pip install -r requirements.txt
* $ source scripts/setenv.sh (Set environment variables like PYTHONPATH
* NB: I added project root to PYTHONPATH.
* Generate results folder by: $ mkdir -p ./results/checkpoint/

### Data: General data processing
* Data spans only 1 day in Nov 2016. Thus, we do not create an out-of-sample test set by time

### Model: General data processing
* When training the model, we might choose only 20% of total data because of memory limitation of my Mac
* We split train and validation sets as 80-20 or 90-10
* Cross validation: We repeat the above step for 5 times to obtain a mean value of selected performance metrics, e.g. rmse
* In validation data, we keep a sparsity of 99%, although data's sparsity is north of 99.9%. see ensure_sparsity(). Reducing sparsity is generally considered a good practice to improve discovery of a recommender. 

### Model: Train the model
* $ python smule/train_model.py --src_data_file=/Users/xin.heng/data/smule/incidences_piano.tsv --nfolds=5 --total_size=0.5 --val_size=0.2
* Best hyperparameter set that I found:
```
    best_model_params = {
        'rank': 15,
        'max_iter': 20,
        'reg_param': 2,
        'alpha': 10
    }
```

### Model: Run the model with the best hyper-parameters
* $ python smule/item_similarities.py --total_size=1.0

### Installation
* Run at project root: $ pip wheel .
* A whl file is created, which can be installed with $ pip install < whl filename >
