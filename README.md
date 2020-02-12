# ChatterBot: Jointly Generative and Selective Transformer Chatbot

## Description

Fundamentally, a generative chatbot can be constructed using the same techniques and architectures as a neural machine translation problem. The conversation is *encoded* into a context, then that context is *decoded* into a new reply which is constructed auto-regressively using a conditioned language model. Previously, the go-to architecture for modeling these seq2seq-style problems was the RNN, then RNN+attention, and now the fully-attentive Transformer. Generative chatbots can produce novel response to novel context if trained on enough data, but are prone to generic responses. Additionally, the language they use is a reflection of their corpus, which can be problematic if trained on a corpus like Reddit text, and they are not well-suited for task-oriented conversations.

The other overarching category of chatbot is the Selective chatbot. This style of chatbot uses similarity scores between an embedded context and a database of candidate responses to choose the most appropriate response. This allows the designer to hand-craft dialogue and control how the bot interacts with users. Unfortunately, this means the conversation will never be original.

Scoring candidate responses is itself an sequence-related problem, and could be concievably framed with a Transformer model acting as the embedding architecture from which to calculate similarity scores. Because the generative and selective chatbot formulations suggest the use of similar models and share a similar foundation in language understanding, I believe the tasks can be learned jointly to create a model that performs better on either task than trained seperately. 

For example, the generative chatbot is essentially a language model, the underpinning of many recent successes in NLP transfer learning. This suggests a better selective chatbot may be built around or with a language model. Conversely, a strictly generative chatbot may be able to ad-lib, but will fail in task-oriented situtations. A hybid model which can choose from novel text and pre-made responses may improve user experience in this situation.

## Architecture

The generative portion of my chatbot follows a typical Transformer neural architecture set-up, with an encoder and decoder block. Recent work on Google's Meena chatbot showed a more powerful decoder is more important than the encoder, so I may try their uneven design. This portion of the model will by trained on the corpus first using crossentropy loss on 8000 classes representing subwords in the corpus.

Next, I plan to build my selective model on top of the generative transformer using the outputs of the encoder and decoder blocks as features. These blocks will already be trained as a language model, so their gradients will be frozen. I am still debating the architecture for the selective network, which must produce two feature vectors from the tranformer's outputs. One option is averaging the generator outputs over time, then running those through dense layers. Another approach might be an attention layer, or even perhaps more transformer layers. This appended network will be trained with triplet loss on common responses.

Additionally, I plan to use relative positional embeddings in the attention layer, which will be dependent on the features of distance between two word representations, as well as speaker embeddings concatenated with word embeddings, so the chatbot may learn multiple roles or personalities.

Lastly, word embeddings were contructed using gensim's word2vec implementation, which featurized 8000 subwords found using Google's sentencepiece unigram encoder implementation. The architecture specifications are shown below:
<br><img src="readme_materials/architecture.png" height=500><br>
Figure 1. Chatbot architecture.<br>

## Data

### Conversation Modeling

The database I am using includes 3 million tweets from conversations between users and customer service accounts for major brands including Spotify, Apple, and Verizon. In long form, this database gives each tweet as a response to one or more other tweets. Furthermore, a tweet may be a response to multiple tweets from different users, or multiple tweets in series from the same user. All this makes Twitter conversations unexpectantly convoluted, and so I took to thinking of conversations as DAGs. Below is an example of interdependencies between tweets represented as a DAG, each edge being a response connection. Nodes A.1 and A.2 are tweets by the same user that were both responsed to by tweet C. My algorithm for generating topological orderings uses breadth-first search from root nodes and flattens layers of the graph to include single-user tweet series. Given the DAG in Figure 2, my algorithm finds the listed conversations chains.
<br><br>
<img src="readme_materials/DAG.png" height="350"><br>
Figure 2. Conversation DAG and topological orderings discovered.
<br>

My BFS algorithm was implemented in Spark SQL to efficiently construct nearly 1 million conversations from the 3 million tweets. The process took less than 5 minutes, so this could easily be expanded to more tweets if I found another database. The sequences this yielded were then broken into 5-tweet sub-sequences for training. At inference time, an entire conversation may be used for context because my relative positional encoding scheme is expandable to any context length, but this is not practical for training when number of examples is more important than learning very long-range dependencies. 

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