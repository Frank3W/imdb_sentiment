# imdb_sentiment

This repo rpesents a data science end-to-end pipline for building and delpoying `Pytorch` sentiment prediction model from IMDB dataset. It is composed of the following steps:

1. Fetch IMBD data - [imdbdata.py](imdbdata.py)
2. Process text - [textproc.py](textproc.py)
3. Encode text into numeric arrays - [imdbdata.py](imdbdata.py)
4. Build Pytorch model - [torchnet.py](torchnet.py)
5. Serve saved NLP operations and model as post API via Flask - [app_sentiment_pred.py](app_sentiment_pred.py)
6. Create a simple front web with form submission to the post API - [index.html](templates/index.html)
