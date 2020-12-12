# imdb_sentiment

This repo presents a data science end-to-end pipeline for building and deploying `Pytorch` NLP sentiment prediction model based on IMDB dataset. The focus is on sequential components along the pipeline not on getting state of the art model itself. The data processing and model training components are organized in decoupled `Python` modules with high reusability and modularization in folder [scr](src).

Specifically, the pipeline is composed of the following steps:

1. Fetch IMBD data - [imdbdata.py](src/imdbdata.py)
2. Process text - [textproc.py](src/textproc.py)
3. Encode text into numeric arrays - [imdbdata.py](src/imdbdata.py)
4. Build Pytorch model - [torchnet.py](src/torchnet.py)
5. Serve trained NLP model as a post API via Flask - [flask_app_sentiment.py](flask_app_sentiment.py)
6. Simple front web with submission form connected to the post API via JQuery Ajax - [index.html](templates/index.html)

The first 3 steps are on data fetching and processing, the step 4 is on model building, and the last 2 steps are on model deployment. It demos the classic 3 main components in a data science pipeline:

**Data Processing** => **Model Building** => **Model Deployment**

The model deployment requires all operations to be reproducible for the purpose of making prediction for the new coming raw data. This covers all data processing steps and trained model prediction. Thus we need to persist the processing operations and trained model, in particularly, for the data-dependent steps.

In Step 2, the data processing operation depends on training data. Specifically, it computes word frequency and then selected top words as vocabulary. Different training data may result in different vocabularies. To address the deployment of this step, class `TextProc` has

- Two working modes: `train`, `eval`
- Load and save functions:  `from_load_wcount_pair`, `from_load_wcount_pair`

It enables loading data processing operations as exact as the ones applied to training data.

Similarly, for Step 4, the persistence of trained model is required. `Pytorch` natively supports persistence of neural network weights but the network topology is not kept in this way. To address that, the model class `FullNet` in module [torchnet.py](src/torchnet.py) has:
- save-load functions for model network topology `from_modeltopology` (alternative class constructor), `save_modeltopology`
- save-load functions for model network weights `save_model_weights`, `load_model_weights`

The notebook [build_model_demo.ipynb](build_model_demo.ipynb) presents a demonstration of the execution of the first 4 steps in the pipeline including the persistence of processing operations and trained models.

To launch the application (which requires saved data processing and model files), run the command

```python
python flask_app_sentiment.py
```
The application keeps all queries with predicted sentiment scores in `SQLite` database at the backend.
