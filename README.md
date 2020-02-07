# ChatterBot: Jointly Generative and Selective Transformer Chatbot
<hr>

## Description
<hr>

Fundamentally, a generative chatbot can be constructed using the same techniques and architectures as a neural machine translation problem. The conversation is *encoded* into a context, then that context is *decoded* into a new reply which is constructed auto-regressively using a conditioned language model. Previously, the go-to architecture for modeling these seq2seq-style problems was the RNN, then RNN+attention, and now the fully-attentive Transformer. Generative chatbots can produce novel response to novel context if trained on enough data, but are prone to generic responses. Additionally, the language they use is a reflection of their corpus, which can be problematic if trained on a corpus like Reddit text, and they are not well-suited for task-oriented conversations.

The other overarching category of chatbot is the Selective chatbot. This style of chatbot uses similarity scores between an embedded context and a database of candidate responses to choose the most appropriate response. This allows the designer to hand-craft dialogue and control how the bot interacts with users. Unfortunately, this means the conversation will never be original.

Scoring candidate responses is itself an sequence-related problem, and could be concievably framed with a Transformer model acting as the embedding architecture from which to calculate similarity scores. Because the generative and selective chatbot formulations suggest the use of similar models and share a similar foundation in language understanding, I believe the tasks can be learned jointly to create a model that performs better on either task than trained seperately. 

For example, the generative chatbot is essentially a language model, the underpinning of many recent successes in NLP transfer learning. This suggests a better selective chatbot may be built around or with a language model. Conversely, a strictly generative chatbot may be able to ad-lib, but will fail in task-oriented situtations. A hybid model which can choose from novel text and pre-made responses may improve user experience in this situation.

## Architecture
<hr>

BERT, or Bidirectional Encoder Representations from Transformers, demonstrated the ability of Transformers to output embeddings for upstream classification layers when induced by a special token in the input sequence, for instance ```[CLS]``` for class. I believe this functionality can be incorporated into a generative seq2seq Transformer to induce the output vector for similarity comparisons. This would allow the transformer to jointly learn generative and selective modeling, essentially training it to rate its own responses.

There are then two training objectvies. First, the selective modeling functionality will be trained using triplet loss, in which the model learns to embed contexts and responses in a joint n-dimensional space in which more appropriate context-response pairs are closer in that space. Second the generative functionality will try to maximize the probability of recreating the ground-truth response given a particular context. These losses may be minimized using the following formula:

L(context, response_true, response_false) = max(0, sim(E(context), D(response_true)) - sim(E(context), D(response_false)) + a) + log P(response_true | context)

where E is the encoder function, D is the decoder function, and a is the triplet loss scalar. The first equation shows the triplet loss on anchor, positive, and negative samples (context, response_true, response_false), while the second equation shows the maximization of the log-likelihood of generating the true response. The architecture is shown below:
<br><br>
<img src="readme_materials/architecture.png" height="500">
<br>

## Progress
<hr>

So far, I have nearly finished implementing my Transformer model in ```model.py```. I am using the vanilla model described in the seminal Vaswani et al. paper, but with the incorporation of learned relative positional encodings from Shaw et al., 2018. 
