The goal is to perform sentiment analysis on the IMDb movie reviews dataset proposed by [Maas et al. (2011)](https://ai.stanford.edu/~amaas/data/sentiment/]).

There are 7 main notebooks:
- data_fetch downloads and structure the dataset
- data_preprocess cleans the text
- data_desc presents descriptive statistics on the reviews
- RoBERTa_SiEBERT_zeroshot performs zero-shot classification with RoBERTa and sentiment analysis with SiEBERT
- RoBERTa_finetuned trains a RoBERTa model on the IMDb dataset
- GPT performs sentiment analysis with ChatGPT 3.5 turbo through the openai API (requires OPENAI_API_KEY to be declared in .env)
- results compares the accuracy of these models

Two models were actually fine-tuned:
- One with a custom classification head, whose results are presented in output/complex
- One with the default classification head, whose results are presented in output/simplex
