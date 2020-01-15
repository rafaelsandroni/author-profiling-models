# Author Profiling 
Author Profiling (AP) is a computational task of recognizing the characteristics of text
authors based on their linguistic patterns. The use of computer computational models allows
us to infer social characteristics from the text, even if the authors do not consciously choose
to place indicators of these characteristics in the text. The AP task can be important
for many practical applications, such as forensic analysis, criminal investigation, and
marketing. Traditional AP approaches often use language knowledge, which requires prior
knowledge and requires manual effort to extract features. Recently, the use of artificial
neural networks has shown satisfactory results in natural language processing (NLP)
problems, however, for author profiling, presents a varied level of success. This paper aims
to organize, define and explore various authorial characterization tasks from the textual
corpus considered, covering three languages (i.e, Portuguese, English and Spanish) and
five textual domains (ie, social networks, questionnaires, SMS etc). Six models based on
neural networks and word embeddings were proposed, performance of models are compared with baseline systems.

# Masters dissertation
[Download masters dissertation latest version](../blob/master/dissertation/dissertation-2019-12-09-vc.pdf)

# Implementation models
Here you can find implemented models with containing both data pipeline and machine learning pipeline.

- **lr_tfidf**: logistic regression + tfidf, /src/models/baseline1

- **cnn_tfidf**: 1D conv net + tfidf, /src/models/baseline2

- **cnn_wv**: multichannel 1D conv net + word vectors, /src/models/baseline3

- **cnn_wv, Kim implementation**: multichannel 1D conv net + word vectors, /src/models/baseline4

- **lstm_wv**: LSTM + word vectors, /baseline5

- **lstm_attention_wv**: LSTM self attention mechanism + word vectors, /src/models/baseline6

- **gru_wv**: GRU + word vectors, /src/models/baseline7

- **cnn_char**: multichannel 1D conv net + char vectors, /src/models/baseline9

- **lstm_attention_char**: LSTM self attention mechanism + char vectors, /src/models/baseline9

# Corpus
Those textual datasets supports 6 author profiling tasks: gender, age, education level, religious, IT formation and politics position, in three languages: portuguese, english and spanish.

This dissertation have structured and defined datasets to author profiling tasks, such as classes distribution and definition of the problems.

- b5-post
- BRMoral
- BlogSet-BR
- Nus-SMS
- The Blog Authorship
- PAN 2013 (PAN-CLEF)

Dataset are splited into stratificated training and test subsets

You can request access to structured datasets to the author.

# Utils evaluation functions
Utils functions build to help implementations, pre-build models, reports etc

/src/functions/

- utils: related to helpers functions
- plot: related to plot functions, using matplotlib and metrics calc
- word vectors: related to embeddings algorithms, training and load pre trained models
- etc


# Reference

@MASTERSDISSERTATION{sandroni-dias,
  title        = "Author profiling from texts using artificial neural networks",
  author       = "Rafael Felipe Sandroni Dias",
  year         = "2019",
  type         = "Master's Dissertation",
  school       = "University of São Paulo",
  address      = "São Paulo, SP, Brazil",
}

