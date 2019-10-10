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
neural networks and word embeddings were proposed, performance of theses models are compared with baseline systems.

# Implementation models

- **lr_tfidf**: logistic regression + tfidf, /baseline1

- **cnn_tfidf**: 1D conv net + tfidf, /baseline2

- **cnn_wv**: multichannel 1D conv net + word vectors, /baseline3

- **cnn_wv, Kim implementation**: multichannel 1D conv net + word vectors, /baseline4

- **lstm_wv**: LSTM + word vectors, /baseline5

- **lstm_attention_wv**: LSTM self attention mechanism + word vectors, /baseline6

- **gru_wv**: GRU + word vectors, /baseline7

- **cnn_char**: multichannel 1D conv net + char vectors, /baseline9

- **lstm_attention_char**: LSTM self attention mechanism + char vectors, /baseline9

# Corpus

Theses textual datasets supports six author profiling tasks: gender, age, education level, religious, IT formation and politics position, in three languages: portuguese, english and spanish.

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

- utils: related to helpers functions
- plot: related to plot functions, using matplotlib and metrics calc
- word vectors: related to embeddings algorithms, training and load pre trained models
- etc

