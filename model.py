#%%
import tensorflow as tf
import numpy as np
import numpy as np
import math
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

#%%

def InterconvertLayerNormLayer():
    return tf.keras.Sequential([
        tf.keras.layers.Activation('linear', dtype = tf.float32),
        tf.keras.layers.LayerNormalization(dtype=tf.float32),
        tf.keras.layers.Activation('linear', dtype = tf.float16),
    ])

class RelativePositionalEncoder(tf.keras.layers.Layer):

    def __init__(self, max_relative_distance, clipping_fn = None, **kwargs):
        super().__init__(**kwargs)
        self.max_relative_distance = max_relative_distance
        if clipping_fn is None:
            self.clipping_fn = lambda x,s : tf.maximum(-s, tf.minimum(s, x))
        else:
            self.clipping_fn = clipping_fn

    def build(self, input_shape):

        assert(len(input_shape)) == 2, 'Input to positional encoder must be a Q, K'
        assert(len(input_shape[0]) == 4), 'Input matrix must be of shape (m, h, t, d)'
        #print(input_shape)
        d = input_shape[0][-1]
        embeddings_shape = (2 * self.max_relative_distance + 1, d)
        #print(embeddings_shape)
        self.embeddings = self.add_weight(shape = embeddings_shape, name = 'positional_embeddings', trainable = True)

        t_q = input_shape[0][-2]
        t_k = input_shape[1][-2]

        t_q_ = tf.expand_dims(tf.range(t_q), -1)
        t_k_ = tf.expand_dims(tf.range(t_k), -1)

        relative_distances = (t_q_ - tf.transpose(t_k_))
        relative_distances = self.clipping_fn(relative_distances, self.max_relative_distance)
        self.relative_distances = relative_distances + tf.minimum(self.max_relative_distance + 1, tf.maximum(t_q, t_k)) - 1
        #Q (m,h,t1,d)
        #a (t1,t2,d)
        #self.pe_matrix = tf.gather(self.pe, relative_distances)

    #pass (Q,K)
    def call(self, X):

        (Q, K) = X

        embeddings = tf.gather(self.embeddings, self.relative_distances)

        embeddings = tf.transpose(embeddings, perm = [0, 2, 1])

        #(m,h,t1,d) dot (t1,d,t2) => (m,h,t1,t2)
        positional_embedding = tf.einsum('mhtd,tds->mhts', Q, embeddings)

        return positional_embedding

#%%
def dot_product_attn(q, k, v, r, len_q, mask = None):
    
    energies = tf.multiply(tf.cast(1/(len_q**0.5), tf.float16), tf.matmul(q, k, transpose_b = True) + r)

    if not mask is None:
        energies = tf.add(tf.multiply(energies, mask), (1. - mask) * -65504)
    
    alphas = tf.nn.softmax(energies, axis = -1)
    
    context = tf.matmul(alphas, v)
    
    return context

#%%


class MultiHeadProjection(tf.keras.layers.Layer):
    
    def __init__(self, projected_dim, heads = 8, **kwargs):
        super().__init__(**kwargs)
        self.h = heads
        self.projected_dim = projected_dim
        
    def build(self, input_shape):
         
        assert(len(input_shape) == 3), 'Expected input of rank 3: (m, Tx, d_model)'
        
        self.m, self.k, self.model_dim = input_shape
        
        self.W = self.add_weight(
                name = 'W',
                shape = (self.h, self.model_dim, self.projected_dim), 
                initializer = 'glorot_normal', 
                trainable = True)
        
    def call(self, X):
        
        X = tf.expand_dims(X, 1) # adds a head layer
        
        output = tf.matmul(X, self.W)
        
        return output

#%%
class AttentionLayer(tf.keras.layers.Layer):
   
    def __init__(self, projected_dim, max_relative_distance = 24, dropout = 0.1, heads = 8, **kwargs):
        super().__init__(**kwargs)
        self.h = heads
        self.projected_dim = projected_dim
        self.dropout_rate = dropout
        self.max_relative_distance = max_relative_distance
        
    def build(self, input_shape):
        
        for input_ in input_shape:
            assert(len(input_) == 3), 'Expected input shape of (m, Tx, d)'
        
        (self.projQ, self.projK, self.projV) = (MultiHeadProjection(self.projected_dim, self.h) 
                                       for input_ in input_shape)
        
        output_d = input_shape[-1][-1]
        
        self.reshaper = tf.keras.layers.Reshape(target_shape = (-1, self.projected_dim * self.h))
        
        self.dense = tf.keras.layers.Dense(output_d, bias_initializer='zeros')

        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)

        self.positional_encoder = RelativePositionalEncoder(self.max_relative_distance)
        

    def call(self, X, mask = None, training = True):
        '''
        Arguments
        X: list of (Q, K, V)
        mask: for softmax layer
        '''
        (Q,K,V) = X
        
        Q, K, V = self.projQ(Q), self.projK(K), self.projV(V)
                
        #print(Q.get_shape(), K.get_shape(), V.get_shape())
        R = self.positional_encoder((Q, K))
        
        attention = dot_product_attn(Q, K, V, R, self.projected_dim, mask = mask)
        
        #print(attention.get_shape())
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        
        flattened = self.reshaper(attention)
       
        output = self.dense(flattened)

        output = self.dropout2(output, training = training)
        
        return output

#%%

class FCNNLayer(tf.keras.layers.Layer):
    
    def __init__(self, d_model, dff, dropout = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.dff = d_model, dff
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(self.dff, activation = 'relu', use_bias = True, bias_initializer='zeros')
        self.dense2 = tf.keras.layers.Dense(self.d_model, activation = 'linear', use_bias = True, bias_initializer='zeros')
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        
    def call(self, X, training = True):
        return self.dropout(self.dense2(self.dense1(X)), training = training)


#%%
class CausalSeperable1DConv(tf.keras.layers.Layer):

    def __init__(self, output_depth, kernel_width, activation = 'relu', **kwargs):
        super().__init__(**kwargs)
        self.output_depth = output_depth
        self.kernel_width = kernel_width
        self.activation = activation

    def build(self,input_shape):
        assert(len(input_shape) == 3), 'Must be (m, Tx, d_model)'
        m, w, nc = input_shape

        self.seperable_conv = tf.keras.layers.SeparableConv2D(self.output_depth, (1, self.kernel_width), 
                strides = 1, padding = 'valid', activation = self.activation, use_bias = False)

        self.padding = tf.keras.layers.ZeroPadding1D((self.kernel_width - 1, 0))

    def call(self, X, conv_mask):

        X = tf.multiply(X, conv_mask)
        X = self.padding(X)
        #m, w, nc -> m, h = 1, w, nc
        X = tf.expand_dims(X, 1)
        X = self.seperable_conv(X)
        X = tf.squeeze(X, axis = 1)
        return X

#%%

class EncoderLayer(tf.keras.layers.Layer):
    
    def __init__(self, dff = 2048, conv_width = (7,9), heads = 8, fcnn_dropout = 0.1, 
            attn_dropout = 0.1, max_relative_distance = 24, **kwargs):
        super().__init__(**kwargs)
        self.h = heads
        self.fcnn_dropout = fcnn_dropout
        self.dff = dff
        self.attn_dropout = attn_dropout
        self.conv_width = conv_width
        self.max_relative_distance = max_relative_distance
        
    def build(self, input_shape):
        assert(len(input_shape) == 3), 'Expected input shape of (m, Tx, d)'
        
        (self.m, self.k, self.d_model) = input_shape
        
        self.projected_dim = self.d_model//self.h

        self.conv1 = CausalSeperable1DConv(self.d_model, self.conv_width[0])
        self.conv2 = CausalSeperable1DConv(self.d_model, self.conv_width[1])
        self.self_attn = AttentionLayer(self.projected_dim, dropout = self.attn_dropout, heads = self.h, max_relative_distance = self.max_relative_distance)
        self.fc = FCNNLayer(self.d_model, self.dff, dropout = self.fcnn_dropout)

        self.layer_norms = [InterconvertLayerNormLayer() for i in range(4)]
        
        self.attn_dropout_layer1 = tf.keras.layers.Dropout(self.attn_dropout)
        self.fcnn_dropout_layer = tf.keras.layers.Dropout(self.fcnn_dropout)
                
    def call(self, X, padding_mask, conv_mask, training = True):
        
        X = X + self.conv1(self.layer_norms[0](X), conv_mask)

        X = X + self.conv2(self.layer_norms[1](X), conv_mask)

        X_attnpath = self.layer_norms[2](X)

        X = X + self.attn_dropout_layer1(self.self_attn([X_attnpath, X_attnpath, X_attnpath], mask = padding_mask, training = training))

        X = X + self.fcnn_dropout_layer(self.fc(self.layer_norms[3](X)))
        
        return X

#%%

class DecoderLayer(EncoderLayer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        super().build(input_shape)
        
        self.encoder_attn = AttentionLayer(self.projected_dim, 
            heads=self.h, 
            dropout= self.attn_dropout, 
            max_relative_distance = self.max_relative_distance)
        self.layer_norms.append(InterconvertLayerNormLayer())

        self.attn_dropout_layer2 = tf.keras.layers.Dropout(self.attn_dropout)
                
    def call(self, X, encoder_output, decoder_mask, encoder_mask, conv_mask, training = True):
        
        X = X + self.conv1(self.layer_norms[0](X), conv_mask)

        X = X + self.conv2(self.layer_norms[1](X), conv_mask)

        X_attnpath = self.layer_norms[2](X)
        # attention mechanism 1
        X = X + self.attn_dropout_layer2(
            self.self_attn([X_attnpath, X_attnpath, X_attnpath], mask = decoder_mask, training = training))

        X = X + self.attn_dropout_layer1(
            self.encoder_attn([self.layer_norms[3](X), encoder_output, encoder_output], mask = encoder_mask, training=training))
        
        X = X + self.fcnn_dropout_layer(self.fc(self.layer_norms[4](X)))
                
        return X

#%%

class HighwayLayer(tf.keras.layers.Layer):

    def __init__(self, highway_dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = highway_dropout

    def build(self, input_shape):

        assert(len(input_shape) == 2), 'Input to highway layer must be (H(X0), X0)'
        assert(input_shape[0] == input_shape[1])
        self.d_model = input_shape[0][-1]
        self.Wt = self.add_weight(shape = (self.d_model, self.d_model), name = 'Wt')
        self.bt = self.add_weight(shape = (1, self.d_model), name = 'bt')
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, X, mask):
        
        (X0, X1) = X 
        Tx = self.dropout(tf.math.multiply(tf.math.sigmoid(tf.matmul(X0, self.Wt) + self.bt), mask))
        return tf.math.multiply(X1, Tx) + tf.math.multiply(X0, (1.0 - Tx))

#%%

class HighwayCNNLayer(tf.keras.layers.Layer):

    def __init__(self, kernel_width, highway_dropout = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.kernel_width = kernel_width
        self.dropout_rate = highway_dropout

    def build(self, input_shape):
        (m, tx, d_model)= input_shape
        self.cnn = CausalSeperable1DConv(d_model, self.kernel_width)
        self.highway = HighwayLayer(self.dropout_rate)

    def call(self, X, mask):
        return self.highway((X, self.cnn(X, mask)), mask)

#%%

class MaskEmbedder(tf.keras.layers.Layer):

    def __init__(self, embedding_layer, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer = embedding_layer

    def call(self, inputs, mask):

        X = self.embedding_layer(inputs)

        loss_mask = tf.cast(self.embedding_layer.compute_mask(inputs), tf.float16)

        padding_mask = loss_mask[:,tf.newaxis,tf.newaxis,:]

        attn_mask = tf.math.multiply(tf.cast(mask, tf.float16), padding_mask)

        return X, attn_mask, padding_mask, tf.expand_dims(loss_mask, -1)

#%%

class RepresentationLayers(tf.keras.layers.Layer):

    def __init__(self, embedding_layer, d_model, num_speakers, num_highway_layers, 
            num_shared_layers, transformer_layer_kwargs, embedding_dropout = 0.1, highway_dropout = 0.1,  **kwargs):

        super().__init__(**kwargs)
        self.embedding_layer = embedding_layer
        self.num_highway_layers = num_highway_layers
        self.num_shared_layers = num_shared_layers
        self.transformer_layer_kwargs = transformer_layer_kwargs
        self.num_speakers = num_speakers
        self.d_model = d_model
        self.highway_dropout = highway_dropout
        self.embedding_dropout = embedding_dropout

    def build(self, input_shape):
        
        self.word_embedder = MaskEmbedder(self.embedding_layer)

        self.highway_layers = [HighwayCNNLayer(7, self.highway_dropout) for i in range(self.num_highway_layers)]

        self.speaker_embedder = tf.keras.layers.Embedding(self.num_speakers + 1, self.d_model, mask_zero = True)
        self.shared_layers = [EncoderLayer(**self.transformer_layer_kwargs) for _ in range(self.num_shared_layers)]
        
        self.context_embedding = self.add_weight(shape = (1,1,self.d_model), name = 'context_embedding')
        self.response_embedding = self.add_weight(shape = (1,1,self.d_model), name= 'response_embedding')

        self.dropout = tf.keras.layers.Dropout(self.embedding_dropout)

    def call(self, X, sender, is_context, attn_precursor_mask, training = True):

        X, attn_mask, padding_mask, conv_mask = self.word_embedder(X, attn_precursor_mask) 

        X = X * tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        X = X + self.speaker_embedder(sender)

        X = tf.cast(X, tf.float16)

        X = 1/3 * tf.multiply(
            conv_mask, 
            tf.cond(tf.constant(is_context, dtype = tf.bool), 
                lambda : tf.add(X, self.context_embedding), 
                lambda : tf.add(X, self.response_embedding))
        )

        X = self.dropout(X)

        for highway_layer in self.highway_layers:
            X = highway_layer(X, conv_mask)

        for i, shared_layer in enumerate(self.shared_layers):
            X = shared_layer(X, attn_mask, conv_mask, training = training)

        return X, attn_mask, padding_mask, conv_mask

class Transformer(tf.keras.Model):

    def __init__(self, 
        num_subwords = 8000, 
        num_speakers = 2, 
        d_model = 512,
        num_encoder_layers = 1, 
        num_decoder_layers = 6, 
        num_highway_layers = 2, 
        num_shared_layers = 4, 
        embedding_dropout = 0.1,
        highway_dropout = 0.1,
        transformer_layer_kwargs = dict(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_subwords = num_subwords
        self.num_speakers = num_speakers
        self.num_decoder_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers
        self.num_shared_layers = num_shared_layers
        self.num_highway_layers = num_highway_layers
        self.d_model = d_model
        self.highway_dropout = highway_dropout
        self.embedding_dropout = embedding_dropout
        self.transformer_layer_kwargs = transformer_layer_kwargs

    def build(self, input_shape):
        assert(len(input_shape) == 4), 'Input must consist of (context, speaker, response, author) set.'
        assert(input_shape[0] == input_shape[1]), 'context and speaker indexes must be same length'
        assert(input_shape[2] == input_shape[3]), 'response and author indexes must be same length'
        (m, tc) = input_shape[0]
        (m, tr) = input_shape[2]

        ## generate masks here
        (m, t_c) = input_shape[0]
        (m, t_r) = input_shape[2]
        
        self.lookahead_mask = tf.linalg.band_part(tf.ones((t_r, t_r)), -1, 0)[tf.newaxis, tf.newaxis, :, :]
        ##
        self.embedding_layer = tf.keras.layers.Embedding(self.num_subwords, self.d_model, mask_zero = True, trainable = False)
 
        self.representation_model = RepresentationLayers(self.embedding_layer, self.d_model, self.num_speakers, self.num_highway_layers, 
            self.num_shared_layers, self.transformer_layer_kwargs, highway_dropout = self.highway_dropout, embedding_dropout = self.embedding_dropout)
        
        self.encoder_layers = [EncoderLayer(**self.transformer_layer_kwargs) for _ in range(self.num_encoder_layers)]

        self.decoder_layers = [DecoderLayer(**self.transformer_layer_kwargs) for _ in range(self.num_decoder_layers)]
    
    @tf.function()
    def encode_context(self, context, sender, training = True):

        context, _, padding_attn_mask, conv_mask = self.representation_model(context, sender, True, self.lookahead_mask, training = training)
        
        for encoder_layer in self.encoder_layers:
            context = encoder_layer(context, padding_attn_mask, conv_mask, training = training)

        return context, padding_attn_mask

    @tf.function()
    def decode_response(self, response, author, context, context_attn_mask, training = True):

        response, lookahead_attn_mask, _, conv_mask = self.representation_model(response, author, False, self.lookahead_mask, training = training)

        for decoder_layer in self.decoder_layers:
            response = decoder_layer(response, context, lookahead_attn_mask, context_attn_mask, conv_mask, training = training)

        response = tf.keras.layers.Activation('linear', dtype = tf.float32)(response)

        output_logits = tf.matmul(response, tf.transpose(self.embedding_layer.embeddings))

        return output_logits, tf.cast(tf.squeeze(conv_mask, axis = -1), tf.float32)

    def call(self, X, training = True):

        (context, sender, response, author) = X

        output_logits, loss_mask = self.decode_response(
                response, 
                author, 
                *self.encode_context(context, sender, training = training), 
                training = training)

        return output_logits, loss_mask

# %%
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, initial_lr, warmup_steps=5000):
        super(CustomSchedule, self).__init__()

        self.warmup_steps = warmup_steps
        self.initial_lr = tf.cast(initial_lr, tf.float32)
    
    def __call__(self, step):
        return tf.cond(step < self.warmup_steps, lambda : self.initial_lr, lambda : tf.math.rsqrt(step) * self.initial_lr)

def TransformerOptimizer(initial_lr):

    adam_opt = tf.keras.optimizers.Adam(CustomSchedule(initial_lr), beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    return mixed_precision.LossScaleOptimizer(adam_opt, loss_scale = 'dynamic')