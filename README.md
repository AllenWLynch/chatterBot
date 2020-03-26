# ChatterBot: Jointly Generative and Selective Transformer Chatbot

## Description

Fundamentally, a generative chatbot can be constructed using the same techniques and architectures as a neural machine translation problem. The conversation is *encoded* into a context, then that context is *decoded* into a new reply which is constructed auto-regressively using a conditioned language model. Previously, the go-to architecture for modeling these seq2seq-style problems was the RNN, then RNN+attention, and now the fully-attentive Transformer. Generative chatbots can produce novel response to novel context if trained on enough data, but are prone to generic responses. Additionally, the language they use is a reflection of their corpus, which can be problematic if trained on a corpus like Reddit text, and they are not well-suited for task-oriented conversations.

The other category of chatbot is the Selective chatbot. This style of chatbot uses similarity scores between an embedded context and a database of candidate responses to choose the most appropriate response. This allows the designer to hand-craft dialogue and control how the bot interacts with users. Unfortunately, this means the conversation will never be original.

Scoring candidate responses is itself an sequence-related problem, and could be concievably framed with a Transformer model acting as the embedding architecture from which to calculate similarity scores. Because the generative and selective chatbot formulations suggest the use of similar models and share a similar foundation in language understanding, I believe the tasks can be learned jointly to create a model that performs better on either task than trained seperately. 

For example, the generative chatbot is essentially a language model, the underpinning of many recent successes in NLP transfer learning. This suggests a better selective chatbot may be built around or with a language model. Conversely, a strictly generative chatbot may be able to ad-lib, but will fail in task-oriented situtations. A hybid model which can choose from novel text and pre-made responses may improve user experience in this situation.

## Architecture

My architecture is based off successful literature models. Specifically, I borrowed the the use of a larger decoder than encoder found to be successful in Google's Meena chatbot, the highway layers and convolutional layers in attention modules of QANet, and the multilayer concatenation used to construct high-dimensional embeddings found in ELMO. 

Inputs to the model consist of context and response sequences, which share the same subword-embedding matrix and causal convolutional highway layers, and learn shared higher-order representations from input subwords. Next, learned speaker embeddings are added to the word representations to mark boundaries in conversations, and perhaps inject substitutable personality into the chatbot. These layered sequence representations are fed into the encoder and decoder layers of the Transformer seq2seq model with learned relative positional encodings, which outputs a distribution over the same shared subword matrix for next word selection.

The selective module of my chatbot takes as input the concatenation of the outputs of the first four layers of the decoder and encoder stacks, which creates a rich, high-dimensional representation of the context and response. As shown in ELMO, concatenation of successive layers of processing yields representations with a higher degree of diversity in the functions being modeled. The selective model is composed of transformer layers, and the output context/response embedding vectors are averages of the transformer output vectors, as in S-BERT, a powerful sentence comparison architecture. These embedding vectors are compared with cosine similarity, with likely context-response pairs being closer in cosine similarity than unlikely pairs. The triplet loss for the selective model and crossentropy loss for the generative model will be trained seperately. First, the generative portion of the model will be trained, during which it will learn meaningful representations for input to the selective model, then the generative model's weights will be frozen and the selective model trained. 

This dual-training scheme will yield complimentary models. The generative model will produce novel, but safe responses, while the selective model can both rate the generator's responses or choose curated responses from a database. Because the generative model supplies a rich, contextual input to the selective model, my selective model may be able to generalize to new response databases and use cases more easily than a selective model that is not joinly trained with a language model.

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

Constructing these conversation trees hit the upper limit of my processing power, but I would like to add more conversations to my training data. Fortunately, another customer support corpus, the Ubuntu dialogue corpus, has a similar structure so I can easily adapt my script to mine its *(context, response)* pairs as well. Unfortunately, It contains over 8 times as much data, so I will need to rent some processing power before I can add that source to my data. 

### Subword Embeddings

The go-to method for representing language in a fixed-length vocab for neural networks is subword embedding, where common words can be represented directly, while rare words can be constructed out of smaller multi-character blocks. This allows the network to learn higher level meanings for many words while also eliminating the out-of-vocab problem when encountering new words. My pre-processing technique for the tweets will follow these steps:

1. Use regex to filter out urls, phone numbers, signatures, usernames, and emojis, replacing some with tokens,
2. Use Sentencepiece encoding for tokenization into subwords. The vocab size, 8000, was taken from Google's Meena which found this to be sufficient for generating quality responses while reducing the model parameters.
3. Encode subwords as index from vocabulary.
4. Append and pad sequences for feeding into network.


### Embedding Layer

I used word2vec to obtain pre-trained word embeddings, which will be frozen during training to reduce model complexity. The same embeddings will be used for the encoder, decoder, and softmax layers of the Transformer to further reduce parameter size.

### Training

To reduce the space needed to train this model, I trained with mixed precision. Weights, loss, and gradients were calculated in fp32, while forward propagation was calculated in fp16. I utilized dynamic loss scaling to prevent gradient underflow. Training metrics were recorded to a Tensorboard dashboard every 50 steps, and the model was evaluated on the test set using the same metrics at the end of every epoch. Additionally, samples of the model's response to test-set contexts were sampled and displayed every epoch. 

On a Azure DSVM instance with a single P40 GPU, I used a batch size of 256, and an initial learning rate of 0.001 that decayed with the inverse square of the step after 10000 warm-up steps.

# Results

Training is ongoing at this moment! I estimate the generative model will need to train for 15 days to make interesting responses, so I am spinning up for training when I don't need my computer for anything else. So far we're at 5 full days of training and chatbot responses are not interesting/thoughtful as of yet. Train and test losses plunge promisingly downward. The problem of short/safe responses is already revealing itself, as the stop token that truncates the response is selected with high probability after one or two words in the response. Ultimately, this generative chatbot may never produce interesting responses, but will provide rich contextual embeddings for the selective portion of the model.

<hr>
Chatbot Samples:<br>
Tweet: So sad how BAD @116988 customer service went from best to worst. India agents have no clue and screw up everything. <br>
Response: <@usr> Why
<br><br>
Tweet: @115850 now confused,same product showing two prices same time. Have already paid 2099 in one of the order thinking I am paying ~978 https://t.co/SyCN5cKH23 <br>
Response: <@usr> Lol
<hr>
<img src="readme_materials/training_progress.png">

#### Catastrophe! A windows update killed my training routine and restarting the model does not lead to further reduction in training loss. This is likely because the ADAM optimizer's weight-specific gradient variables were not saved, so the optimizer lost momentum and is stuck in some minima. I'll have to retrain from scratch.

# Conversation Strategy

Once the generative model is done training, I could use a number of sampling techniques to generate high quality responses. The main requirement of the chosen sampling technique is that it must promote longer responses than I am observing from the model right now. 

I plan to test and compare two response-generating strategies:
1. Beam Search -> Concurrently grows multiple hypotheses to maximize P(Y|X)
2. Sample and Rank -> Recently shown to produce good results with Google's Meena chatbot. A simple method where many replies are generated, then ranked on some non-differentiable metric to choose the best response.