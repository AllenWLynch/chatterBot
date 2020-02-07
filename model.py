#%%
import tensorflow as tf
import numpy as np
import numpy as np
import math
#%%
# # Scaled Dot-Product attention function

def dot_product_attn(q, k, v, len_q, additional_encoding, mask = None):
    
    energies = tf.multiply(1/(len_q**0.5), tf.matmul(q, k, transpose_b = True) + additional_encoding)
    
    if not mask is None:
        mask = (1. - mask) * -1e9
        energies = tf.add(energies, mask)
    
    alphas = tf.nn.softmax(energies, axis = -1)
    
    context = tf.matmul(alphas, v)
    
    return context


# # Multihead Projection

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



class AttentionLayer(tf.keras.layers.Layer):
   
    def __init__(self, projected_dim, positional_encoder, dropout = 0.1, heads = 8, **kwargs):
        super().__init__(**kwargs)
        self.h = heads
        self.projected_dim = projected_dim
        self.dropout_rate = dropout
        self.positional_encoder
        
    def build(self, input_shape):
        
        for input_ in input_shape:
            assert(len(input_) == 3), 'Expected input shape of (m, Tx, d)'
        
        (self.projQ, self.projK, self.projV) = (MultiHeadProjection(self.projected_dim, self.h) 
                                       for input_ in input_shape)
        
        output_d = input_shape[-1][-1]
        
        self.reshaper = tf.keras.layers.Reshape(target_shape = (-1, self.projected_dim * self.h))
        
        self.dense = tf.keras.layers.Dense(output_d)

        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        
    def call(self, X, mask = None, training = True):
        '''
        Arguments
        X: list of (Q, K, V)
        mask: for softmax layer
        '''
        (Q,K,V) = X
        
        Q, K, V = self.projQ(Q), self.projK(K), self.projV(V)
        
        #print(Q.get_shape(), K.get_shape(), V.get_shape())
        RPEs = self.positional_encoder(Q, K)
        
        attention = dot_product_attn(Q, K, V, self.projected_dim, RPEs, mask = mask)
        
        #print(attention.get_shape())
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        
        flattened = self.reshaper(attention)
       
        output = self.dense(flattened)

        output = self.dropout2(output, training = training)
        
        return output


# # Fully Connected Layer

class FCNNLayer(tf.keras.layers.Layer):
    
    def __init__(self, d_model, dff, dropout = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.dff = d_model, dff
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(self.dff, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(self.d_model, activation = 'linear')
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        
    def call(self, X, training = True):
        return self.dropout(self.dense2(self.dense1(X)), training = training)


# #  Encoder Layer

class TransformerEncoder(tf.keras.layers.Layer):
    
    def __init__(self, dff = 1024, heads = 8, fcnn_dropout = 0.1, attn_dropout = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.h = heads
        self.fcnn_dropout = fcnn_dropout
        self.dff = dff
        self.attn_dropout = attn_dropout
        
    def build(self, input_shape):
        assert(len(input_shape) == 3), 'Expected input shape of (m, Tx, d)'
        
        (self.m, self.k, self.d_model) = input_shape
        
        self.projected_dim = self.d_model//self.h
        
        self.attn = AttentionLayer(self.projected_dim, dropout = self.attn_dropout, heads = self.h)
        self.norm1 = tf.keras.layers.LayerNormalization()
        
        self.fcnn = FCNNLayer(self.d_model, self.dff, dropout = self.fcnn_dropout)
        self.norm2 = tf.keras.layers.LayerNormalization()
                
    def call(self, X, rel_pos_encoder, training = True, mask = None):
        
        attn_output = self.attn([X,X,X], rel_pos_encoder, mask = mask, training=training)
        
        X = self.norm1(attn_output + X)
        
        fcnn_output = self.fcnn(X, training= training)
        
        X = self.norm2(fcnn_output + X)
        
        return X

class TransformerDecoder(TransformerEncoder):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        super().build(input_shape)
        
        self.intr_attn = AttentionLayer(self.projected_dim, heads=self.h, dropout= self.attn_dropout)
        self.norm0 = tf.keras.layers.LayerNormalization()
                
    def call(self, X, encoder_output, rel_pos_encoder, decoder_mask = None, encoder_mask = None, training = True):
        
        # attention mechanism 1
        attn_output = self.intr_attn([X,X,X], rel_pos_encoder, mask = decoder_mask, training=training)
        X = self.norm0(attn_output + X)
                              
        # attention mechanism 2
        attn_output = self.attn([X,encoder_output, encoder_output], rel_pos_encoder, mask = encoder_mask, training=training)
        X = self.norm1(attn_output + X)
                                 
        # fcnn
        fcnn_output = self.fcnn(X, training = training)
        X = self.norm2(fcnn_output + X)               
                
        return X


# # Position Encoding Layer
class RelativePositionalEncoder(tf.keras.layers.Layer):

    def __init__(self, max_relative_distance, clipping_fn = None, **kwargs):
        super().__init__(**kwargs)
        self.max_relative_distance = max_relative_distance
        if clipping_fn is None:
            self.clipping_fn = lambda x,s : tf.maximum(-s, tf.minimum(s, x))
        else:
            self.clipping_fn = clipping_fn

    def build(self, input_shape):
        
        assert(len(input_shape)) == 4, 'Input to positional encoder must be a attention-matrix (Q,K,or V) with shape: (m, h, tx, d)'
        #print(input_shape)

        embeddings_shape = (2 * (self.max_relative_distance - 1) + 1, d)
        #print(embeddings_shape)
        self.pe = self.add_weight(shape = embeddings_shape, name = 'positional_embeddings')
        #self.pe_matrix = tf.gather(self.pe, relative_distances)

    def call(self, attnQ, attnK):
        t1 = attnQ.get_shape()[-2]
        t2 = attnK.get_shape()[-2]

        t1_ = tf.expand_dims(tf.range(t1), -1)
        t2_ = tf.expand_dims(tf.range(t2), -1)

        relative_distances = (t1_ - tf.transpose(t2_))
        relative_distances = self.clipping_fn(relative_distances, self.max_relative_distance)
        relative_distances = relative_distances + tf.min(self.max_relative_distance + 1, tf.maximum(t1, t2)) - 1
        #Q (m,h,t1,d)
        #a (t1,t2,d)
        embeddings = tf.gather(self.pe, relative_distances)
        #a(t1,d,t2)
        embeddings = tf.transpose(embeddings, perm = [0,2,1])
        #(m,h,t1,d) dot (t1,d,t2) => (m,h,t1,t2)
        RPEs = tf.einsum('mhtd,tds->mhts', attnQ, embeddings)
        return RPEs


class SpeakerEncoder(tf.keras.layers.Layer):

    def __init__(self, num_speakers, **kwargs):
        super().__init__(**kwargs)
        

    def build(self, input_shape):
        
        assert(len(input_shape)) == 4, 'Input to positional encoder must be a attention-matrix (Q,K,or V) with shape: (m, h, tx, d)'
        #print(input_shape)

        embeddings_shape = (2 * (self.max_relative_distance - 1) + 1, d)
        #print(embeddings_shape)
        self.pe = self.add_weight(shape = embeddings_shape, name = 'positional_embeddings')
        #self.pe_matrix = tf.gather(self.pe, relative_distances)

    def call(self, attnQ, attnK):
        t1 = attnQ.get_shape()[-2]
        t2 = attnK.get_shape()[-2]

        t1_ = tf.expand_dims(tf.range(t1), -1)
        t2_ = tf.expand_dims(tf.range(t2), -1)

        relative_distances = (t1_ - tf.transpose(t2_))
        relative_distances = self.clipping_fn(relative_distances, self.max_relative_distance)
        relative_distances = relative_distances + tf.min(self.max_relative_distance + 1, tf.maximum(t1, t2)) - 1
        #Q (m,h,t1,d)
        #a (t1,t2,d)
        embeddings = tf.gather(self.pe, relative_distances)
        #a(t1,d,t2)
        embeddings = tf.transpose(embeddings, perm = [0,2,1])
        #(m,h,t1,d) dot (t1,d,t2) => (m,h,t1,t2)
        RPEs = tf.einsum('mhtd,tds->mhts', attnQ, embeddings)
        return RPEs
# # Encoder Stack

def convert_embedding_mask(embedding_mask):
    return self.embedding.compute_mask(seqs)[:, tf.newaxis, tf.newaxis, :]


##### ACTUALLY BUILD YOUR LAYERS HERE?
class EncoderStack(tf.keras.layers.Layer):
    
    def __init__(self, num_classes, d_model = 512, num_layers = 6, num_heads = 8, attn_dropout = 0.1, fcnn_dropout = 0.1, dff = 1024, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.fcnn_dropout = fcnn_dropout
        self.attn_dropout = attn_dropout
        self.dff = dff
        
    def build(self, input_shape):

        self.embedding_dropout = tf.keras.layers.Dropout(self.dropout)
        
        self.encoders = [
            TransformerEncoder(dff = self.dff, heads = self.num_heads, fcnn_dropout = self.fcnn_dropout, attn_dropout=self.attn_dropout) 
            for i in range(self.num_layers)
        ]
        
    def call(self, embedded_seqs, embedding_mask, rel_pos_encoder, training = True):
        
        #expand the mask from the embedding layer from (m, Tx) to (m, 1, 1, Tx) for multihead softmax
        encoder_mask = tf.dtypes.cast(convert_embedding_mask(embedding_mask), 'float32')

        X = self.embedding_dropout(embedded_seqs, training = training)
        
        #call(self, X, encoder_output, lookahead_mask = None, encoder_padding_mask = None, training = True)
        for encoder in self.encoders:
            X = encoder(X, rel_pos_encoder, mask = encoder_mask, training = training)
            
        return X, encoder_mask

# # Decoder Stack

class DecoderStack(tf.keras.layers.Layer):
    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
                
        self.embedding_dropout = tf.keras.layers.Dropout(self.dropout)
        
        self.decoders = [
            TransformerDecoder(dff = self.dff, heads = self.num_heads, fcnn_dropout = self.fcnn_dropout, attn_dropout=self.attn_dropout) 
            for i in range(self.num_layers)
        ]

        self.lookahead_mask = None
        #calculate trailing mask here
            
    def call(self, embedded_seqs, decoder_embedding_mask, encoder_output, encoder_mask, rel_pos_encoder, training = True):

        loss_mask = convert_embedding_mask(decoder_embedding_mask)
        #expand the mask from the embedding layer from (m, Tx) to (m, 1, 1, Tx) for multihead softmax
        decoder_mask = tf.dtypes.cast(loss_mask, 'float32') * self.lookahead_mask

        X = self.embedding_dropout(embedded_seqs, training = training)
        
        #call(self, X, encoder_output, lookahead_mask = None, encoder_padding_mask = None, training = True)
        for decoder in self.decoders:
            X = decoder(X, encoder_output, rel_pos_encoder, decoder_mask = decoder_mask, 
                        encoder_mask = encoder_mask, training = training)
            
        return X, loss_mask
        

# # Transformer Model

class TransformerModel(tf.keras.Model):

    def __init__(self, num_encoder_classes, num_decoder_classes, d_model = 512, num_layers = 6, num_heads = 8, dropout = 0.1, dff = 2048):
        super().__init__()

        self.encoder_stack = EncoderStack(num_encoder_classes, d_model, num_layers, num_heads, dropout, dff)

        self.decoder_stack = DecoderStack(num_decoder_classes, d_model, num_layers, num_heads, dropout, dff)

    def call(self, X, Y, training):

            enc_output, encoder_mask = self.encoder_stack(X, training = training)
        
            logits, mask = self.decoder_stack((Y, enc_output, encoder_mask), training = training)

            return logits, mask
        


class Transformer():

    def __init__(self, num_encoder_classes, num_decoder_classes, d_model = 512, num_layers = 6, num_heads = 8, dropout = 0.1, dff = 2048):

        self.model = TransformerModel(num_encoder_classes, num_decoder_classes, d_model, num_layers, num_heads, dropout, dff)

        self.loss = TransformerLoss()

        self.opt = TransformerOptimizer(d_model)


    def train_step(self, X,Y, train = True):

        decoder_input = Y[:,:-1] # don't include end token pushed into decoder
        decoder_target = Y[:,1:] # don't include start token in decoder output label

        with tf.GradientTape() as tape:

            predictions, mask = self.model(X, decoder_input, training = train)

            loss = self.loss(decoder_target, predictions, mask)

        gradients = tape.gradient(loss, self.model.trainable_variables)    
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss.numpy()


# %%
