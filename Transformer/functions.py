import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Embedding, Dense, Input, Dropout, LayerNormalization, Conv1D, Reshape

def scaled_dot_product_attention(Q, K, V, mask=None):

    scaled_attention_logits = tf.matmul(Q,K, transpose_b = True)/np.sqrt(K.shape[-1])

    if mask is not None: 
        scaled_attention_logits += (1. - mask) *(-1e9)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights,V)
    
    return output, attention_weights



    
