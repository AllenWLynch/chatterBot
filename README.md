# ChatterBot: Jointly Generative and Selective Transformer Chatbot

## Description

Fundamentally, a generative chatbot can be constructed using the same techniques and architectures as a neural machine translation problem. The conversation is *encoded* into a context, then that context is *decoded* into a new reply which is constructed auto-regressively using a conditioned language model. Previously, the go-to architecture for modeling these seq2seq-style problems was the RNN, then RNN+attention, and now the fully-attentive Transformer. Generative chatbots can produce novel response to novel context if trained on enough data, but are prone to generic responses. Additionally, the language they use is a reflection of their corpus, which can be problematic if trained on a corpus like Reddit text, and they are not well-suited for task-oriented conversations.

The other overarching category of chatbot is the Selective chatbot. This style of chatbot uses similarity scores between an embedded context and a database of candidate responses to choose the most appropriate response. This allows the designer to hand-craft dialogue and control how the bot interacts with users. Unfortunately, this means the conversation will never be original.

Scoring candidate responses is itself an sequence-related problem, and could be concievably framed with a Transformer model acting as the embedding architecture from which to calculate similarity scores. Because the generative and selective chatbot formulations suggest the use of similar models and share a similar foundation in language understanding, I believe the tasks can be learned jointly to create a model that performs better on either task than trained seperately. 

For example, the generative chatbot is essentially a language model, the underpinning of many recent successes in NLP transfer learning. This suggests a better selective chatbot may be built around or with a language model. Conversely, a strictly generative chatbot may be able to ad-lib, but will fail in task-oriented situtations. A hybid model which can choose from novel text and pre-made responses may improve user experience in this situation.

## Architecture

My architecture extends the successful QANet architecture, which relies on a shared representation module for the context and response (since they are the same language in QA and chatbot applications), followed by a decoder module to produce distributions over the embeddings for softmax logits. The shared embeddings, CNN+highway, which learns common n-grams from subword embeddings, and low-level transformer layers reduce parameters needed to perform similar functions in the context and response flows.

In addition to the generative module, my selective model will also use the shared representation module as input. The selective model will be composed of transformer layers, and the output context/response embedding vectors will be averages of the transformer output vectors, as in S-BERT, a powerful sentence comparison architecture. These embedding vectors will be compared with cosine similarity, with likely context-response pairs being closer in cosine similarity than unlikely pairs. The triplet loss for the selective model and crossentropy loss for the generative model will be trained jointly, with a *(context, response)* pair sampled from the dataset used as a positive example, and a negative sample selected from a buffer of past samples.

Additionally, I plan to use relative positional embeddings in the attention layer, which will be dependent on the features of distance between two word representations, as well as speaker embeddings concatenated with word embeddings, so the chatbot may learn multiple roles or personalities.

Lastly, word embeddings were contructed using gensim's word2vec implementation, which featurized 8000 subwords found using Google's sentencepiece unigram encoder implementation. The architecture specifications are shown below:
<br><img src="readme_materials/architecture.png" width=750><br>
Figure 1. Chatbot architecture.<br>

## Data

### Conversation Modeling

The database I am using includes 3 million tweets from conversations between users and customer service accounts for major brands including Spotify, Apple, and Playsation. In long form, this database gives each tweet as a response to one or more other tweets. Furthermore, a tweet may be a response to multiple tweets from different users, or multiple tweets in series from the same user. All this makes Twitter conversations unexpectantly convoluted, and so I took to thinking of conversations as DAGs. Below is an example of interdependencies between tweets represented as a DAG, each edge being a response connection. Nodes A.1 and A.2 are tweets by the same user that were both responded to by tweet C. My algorithm for generating topological orderings uses breadth-first with one-step lookahead from root nodes and flattens layers of the graph to include single-user tweet series. Given the DAG in Figure 2, my algorithm finds the listed conversations chains, while merging multi-message-one-user events into agglomerated messages.

<br><br>
<img src="readme_materials/DAG.png" height="350"><br>
Figure 2. Conversation DAG and topological orderings discovered.
<br>

The algorithm was implemented in Spark SQL to efficiently construct nearly 1.7 million context-response pairs from the 3 million tweets. The process took less than 5 minutes, so this could easily be expanded to more tweets if I found another twitter support dataset. A breakdown of conversations mined shows that conversations with one response make up the majority of conversations. That pattern is explained by support agents frequently requesting the user send them a direct message, then their conversation leaving the record. Plotted with log scale to show all frequencies, an interesting pattern emerges: the frequencies are often grouped in pairs of two. What this shows is that the user requesting support is most likely to end the conversation, since users respond at length = 0, 2, 4, 6, etc. This is likely because they get the help they need, then thank the support agent to end the conversation.

<img src="readme_materials/frequency.png" height=350>
<img src="readme_materials/log_freq.png" height=350><br>
Figure 3. Length of conversations in the dataset.

### Subword Embeddings

The go-to method for representing language in a fixed-length vocab for neural networks is subword embedding, where common words can be represented directly, while rare words can be constructed out of smaller multi-character blocks. This allows the network to learn higher level meanings for many words while also eliminating the out-of-vocab problem when encountering new words. My pre-processing technique for the tweets will follow these steps:

1. Use regex to filter out urls, phone numbers, signatures, usernames, and emojis, replacing some with tokens,
2. Use Sentencepiece encoding for tokenization into subwords. The vocab size, 8000, was taken from Google's Meena which found this to be sufficient for generating quality responses while reducing the model parameters.
3. Encode subwords as index from vocabulary.
4. Append and pad sequences for feeding into network.


### Embedding Layer
I used word2vec to obtain pre-trained word embeddings, which will be frozen during training to reduce model complexity. The same embeddings will be used for the encoder, decoder, and softmax layers of the Transformer to further reduce parameter size.

### Progress
* Finished programming architecture
* Finished mining for conversations, represented as lists of tweet IDs
* Finished training subword encoder
* Finished pretrained word embeddings
* To Do:
  * Finish data input pipeline
  * Train network. Probably going to rent cloud time because it takes a while to train a language model.

