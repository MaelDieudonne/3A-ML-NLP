Project for the [Machine Learning for Natural Language Processing](https://www.ensae.fr/courses/4237) course at Ensae

The goal is to perform sentiment analysis on the IMDb movie reviews dataset propose by [Maas et al. (2011)](https://ai.stanford.edu/~amaas/data/sentiment/]).

There are 7 notebooks:
- data_fetch downloads and structure the dataset
- data_preprocess cleans the text
- data_desc presents descriptive statistics on the reviews
- RoBERTa_SiEBERT_zeroshot performs zero-shot classification with RoBERTa and sentiment analysis with SiEBERT
- RoBERTa_finetuned trains a RoBERTa model on the IMDb dataset
- GPT performs sentiment analysis with ChatGPT 3.5 turbo through the openai API
- results compares the accuracy of these models
