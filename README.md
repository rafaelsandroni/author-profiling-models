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
neural networks and Wword embeddings were proposed, compared with baseline systems.

# Implementation models

- Baseline1 (lr_tfidf): tfidf + logistic regression

### baseline2 (cnn_tfidf): 1D conv net + tfidf

### baseline3 (cnn_wv): multichannel 1D conv net + word vectors

### baseline4 (cnn_wv, Kim hyperparameters): multichannel word vectors + 1D conv net

### baseline5 (lstm_attention_wv):: LSTM self attention mechanism + word vectors

### baseline9 (cnn_char):: multichannel 1D conv net + char vectors

### baseline9 (lstm_attention_char):: LSTM self attention mechanism + char vectors

# functions

-- utils
-- plot
-- word vectors
-- etc

