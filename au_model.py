import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True

from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Activation, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.layers import Reshape, Permute, Multiply, Add, Concatenate
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import numpy as np

# other keras related library
from keras_pos_embd import PositionEmbedding, TrigPosEmbedding
from keras_layer_normalization import LayerNormalization
from keras_multi_head import MultiHeadAttention
from keras_position_wise_feed_forward import FeedForward
from keras_embed_sim import EmbeddingRet, EmbeddingSim

import math
from tqdm import tqdm

import math
import numpy as np

AU_count =12

AU_Mapping = {0:'Inner Brow Raiser',1:'Outer Brow Raiser',
        2:'Brow Lowerer',3:'Upper Lid Raiser',
        4:'Cheek Raiser', 5:'Lid Tightener',
        6:'Nose Wrinkler', 7: 'Upper Lip Raiser', 
        8:'Lip Corner Puller',9:'Dimpler',10:'Lip Corner Depressor', 
        11:'Chin Raiser', 12: 'Lip Stretcher',13:'Lip Tightener',
        14:'Lip pressor',15:'Lips Part',16:'Jaw Drop',17:'Eyes Closed'}

def get_expression_confidence_scores(au_indexes, threshold = 0.3):
    # reference https://link.springer.com/article/10.1007/s12652-019-01278-2
    au_set = set([AU_Mapping[index] for index in au_indexes])
    
    scores = {
        'happy': 0,
        'surprise': 0,
        'angry': 0,
        'neutral': 0
    }
    
    # happy AU
    if ('Cheek Raiser' in au_set):
        scores['happy'] += 0.4
    if ('Lip Corner Puller' in au_set):
        scores['happy'] += 0.6
    
    # surprised AU
    if ('Jaw Drop' in au_set):
        scores['surprise'] += 0.4
    if ('Inner Brow Raiser' in au_set):
        scores['surprise'] += 0.2
    if ('Outer Brow Raiser' in au_set):
        scores['surprise'] += 0.2
    if ('Upper Lid Raiser' in au_set):
        scores['surprise'] += 0.2

    # "angry" AU
    if ('Nose Wrinkler' in au_set):
        scores['angry'] += 0.4
    if ('Lip Corner Depressor' in au_set):
        scores['angry'] += 0.2
    if ('Brow Lowerer' in au_set):
        scores['angry'] += 0.1
    if ('Lip Tightener' in au_set):
        scores['angry'] += 0.1
    if ('Upper Lid Raiser' in au_set):
        scores['angry'] += 0.1
    if ('Lip Tightener' in au_set):
        scores['angry'] += 0.1

    return scores
    # # find max score
    # scores = [happy_score, surprised_score, angry_score]
    # labels = ('happy', 'surprise', 'angry')
    # max_score = scores[np.argmax(scores)]
    # if max_score >= threshold:
    #     return labels[np.argmax(scores)]
    # return 'neutral'

def gelu(x):
    """An approximation of gelu.
    See: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1.0 + tf.math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


__all__ = [
    'get_custom_objects', 'get_encoders', 'get_decoders', 'get_model', 'decode',
    'attention_builder', 'feed_forward_builder', 'get_encoder_component', 'get_decoder_component',
]


def get_custom_objects():
    return {
        'gelu': gelu,
        'LayerNormalization': LayerNormalization,
        'MultiHeadAttention': MultiHeadAttention,
        'FeedForward': FeedForward,
        'TrigPosEmbedding': TrigPosEmbedding,
        'EmbeddingRet': EmbeddingRet,
        'EmbeddingSim': EmbeddingSim,
    }


def _wrap_layer(name,
        input_layer,
        build_func,
        dropout_rate=0.0,
        trainable=True):
    """Wrap layers with residual, normalization and dropout.
    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param build_func: A callable that takes the input tensor and generates the output tensor.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :return: Output layer.
    """
    build_output = build_func(input_layer)
    if dropout_rate > 0.0:
        dropout_layer = Dropout(
            rate=dropout_rate,
            name='%s-Dropout' % name,
        )(build_output)
    else:
        dropout_layer = build_output
    if isinstance(input_layer, list):
        input_layer = input_layer[0]
    add_layer = Add(name='%s-Add' % name)([input_layer, dropout_layer])
    normal_layer = LayerNormalization(
        trainable=trainable,
        name='%s-Norm' % name,
    )(add_layer)
    return normal_layer



def attention_builder(name,
                      head_num,
                      activation,
                      history_only,
                      trainable=True):
    """Get multi-head self-attention builder.
    :param name: Prefix of names for internal layers.
    :param head_num: Number of heads in multi-head self-attention.
    :param activation: Activation for multi-head self-attention.
    :param history_only: Only use history data.
    :param trainable: Whether the layer is trainable.
    :return:
    """
    def _attention_builder(x):
        return MultiHeadAttention(
            head_num=head_num,
            activation=activation,
            history_only=history_only,
            trainable=trainable,
            name=name,
        )(x)
    return _attention_builder


def feed_forward_builder(name,
                         hidden_dim,
                         activation,
                         trainable=True):
    """Get position-wise feed-forward layer builder.
    :param name: Prefix of names for internal layers.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param activation: Activation for feed-forward layer.
    :param trainable: Whether the layer is trainable.
    :return:
    """
    def _feed_forward_builder(x):
        return FeedForward(
            units=hidden_dim,
            activation=activation,
            trainable=trainable,
            name=name,
        )(x)
    return _feed_forward_builder


def get_encoder_component(name,
              input_layer,
              head_num,
              hidden_dim,
              attention_activation=None,
              feed_forward_activation=gelu,
              dropout_rate=0.0,
              trainable=True,):
    """Multi-head self-attention and feed-forward layer.
    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :return: Output layer.
    """
    attention_name = '%s-MultiHeadSelfAttention' % name
    feed_forward_name = '%s-FeedForward' % name
    attention_layer = _wrap_layer(
        name=attention_name,
        input_layer=input_layer,
        build_func=attention_builder(
            name=attention_name,
            head_num=head_num,
            activation=attention_activation,
            history_only=False,
            trainable=trainable,
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
    )
    feed_forward_layer = _wrap_layer(
        name=feed_forward_name,
        input_layer=attention_layer,
        build_func=feed_forward_builder(
            name=feed_forward_name,
            hidden_dim=hidden_dim,
            activation=feed_forward_activation,
            trainable=trainable,
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
    )

    return feed_forward_layer


def get_decoder_component(name,
              input_layer,
              encoded_layer,
              head_num,
              hidden_dim,
              attention_activation=None,
              feed_forward_activation=gelu,
              dropout_rate=0.0,
              trainable=True):
    """Multi-head self-attention, multi-head query attention and feed-forward layer.
    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param encoded_layer: Encoded layer from encoder.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :return: Output layer.
    """
    self_attention_name = '%s-MultiHeadSelfAttention' % name
    query_attention_name = '%s-MultiHeadQueryAttention' % name
    feed_forward_name = '%s-FeedForward' % name
    self_attention_layer = _wrap_layer(
        name=self_attention_name,
        input_layer=input_layer,
        build_func=attention_builder(
            name=self_attention_name,
            head_num=head_num,
            activation=attention_activation,
            history_only=True,
            trainable=trainable,
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
    )
    query_attention_layer = _wrap_layer(
        name=query_attention_name,
        input_layer=[self_attention_layer, encoded_layer, encoded_layer],
        build_func=attention_builder(
            name=query_attention_name,
            head_num=head_num,
            activation=attention_activation,
            history_only=False,
            trainable=trainable,
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
    )
    feed_forward_layer = _wrap_layer(
        name=feed_forward_name,
        input_layer=query_attention_layer,
        build_func=feed_forward_builder(
            name=feed_forward_name,
            hidden_dim=hidden_dim,
            activation=feed_forward_activation,
            trainable=trainable,
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
    )
    return feed_forward_layer


def get_encoders(name,encoder_num,
          input_layer,
          head_num,
          hidden_dim,
          attention_activation=None,
          feed_forward_activation=gelu,
          dropout_rate=0.0,
          trainable=True):
    """Get encoders.
    :param encoder_num: Number of encoder components.
    :param input_layer: Input layer.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :return: Output layer.
    """
    last_layer = input_layer
    for i in range(encoder_num):
        last_layer = get_encoder_component(
            name='%sEncoder-%d' % (name,i + 1),
            input_layer=last_layer,
            head_num=head_num,
            hidden_dim=hidden_dim,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
            trainable=trainable,
        )
    return last_layer


def get_decoders(name,decoder_num,
                 input_layer,
                 encoded_layer,
                 head_num,
                 hidden_dim,
                 attention_activation=None,
                 feed_forward_activation=gelu,
                 dropout_rate=0.0,
                 trainable=True):
    """Get decoders.
    :param decoder_num: Number of decoder components.
    :param input_layer: Input layer.
    :param encoded_layer: Encoded layer from encoder.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :return: Output layer.
    """
    last_layer = input_layer
    for i in range(decoder_num):
        last_layer = get_decoder_component(
            name='%sDecoder-%d' % (name,i + 1),
            input_layer=last_layer,
            encoded_layer=encoded_layer,
            head_num=head_num,
            hidden_dim=hidden_dim,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
            trainable=trainable,
        )
    return last_layer


def _get_max_suffix_repeat_times(tokens, max_len):
    detect_len = min(max_len, len(tokens))
    next = [-1] * detect_len
    k = -1
    for i in range(1, detect_len):
        while k >= 0 and tokens[len(tokens) - i - 1] != tokens[len(tokens) - k - 2]:
            k = next[k]
        if tokens[len(tokens) - i - 1] == tokens[len(tokens) - k - 2]:
            k += 1
        next[i] = k
    max_repeat = 1
    for i in range(2, detect_len):
        if next[i] >= 0 and (i + 1) % (i - next[i]) == 0:
            max_repeat = max(max_repeat, (i + 1) // (i - next[i]))
    return max_repeat


def decode(model,
      tokens,
      start_token,
      end_token,
      pad_token,
      top_k=1,
      temperature=1.0,
      max_len=10000,
      max_repeat=10,
      max_repeat_block=10):
    """Decode with the given model and input tokens.
    :param model: The trained model.
    :param tokens: The input tokens of encoder.
    :param start_token: The token that represents the start of a sentence.
    :param end_token: The token that represents the end of a sentence.
    :param pad_token: The token that represents padding.
    :param top_k: Choose the last token from top K.
    :param temperature: Randomness in boltzmann distribution.
    :param max_len: Maximum length of decoded list.
    :param max_repeat: Maximum number of repeating blocks.
    :param max_repeat_block: Maximum length of the repeating block.
    :return: Decoded tokens.
    """
    is_single = not isinstance(tokens[0], list)
    if is_single:
        tokens = [tokens]
    batch_size = len(tokens)
    decoder_inputs = [[start_token] for _ in range(batch_size)]
    outputs = [None for _ in range(batch_size)]
    output_len = 1
    while len(list(filter(lambda x: x is None, outputs))) > 0:
        output_len += 1
        batch_inputs, batch_outputs = [], []
        max_input_len = 0
        index_map = {}
        for i in range(batch_size):
            if outputs[i] is None:
                index_map[len(batch_inputs)] = i
                batch_inputs.append(tokens[i][:])
                batch_outputs.append(decoder_inputs[i])
                max_input_len = max(max_input_len, len(tokens[i]))
        for i in range(len(batch_inputs)):
            batch_inputs[i] += [pad_token] * (max_input_len - len(batch_inputs[i]))
        predicts = model.predict([np.array(batch_inputs), np.array(batch_outputs)])
        for i in range(len(predicts)):
            if top_k == 1:
                last_token = predicts[i][-1].argmax(axis=-1)
            else:
                probs = [(prob, j) for j, prob in enumerate(predicts[i][-1])]
                probs.sort(reverse=True)
                probs = probs[:top_k]
                indices, probs = list(map(lambda x: x[1], probs)), list(map(lambda x: x[0], probs))
                probs = np.array(probs) / temperature
                probs = probs - np.max(probs)
                probs = np.exp(probs)
                probs = probs / np.sum(probs)
                last_token = np.random.choice(indices, p=probs)
            decoder_inputs[index_map[i]].append(last_token)
            if last_token == end_token or\
                    (max_len is not None and output_len >= max_len) or\
                    _get_max_suffix_repeat_times(decoder_inputs[index_map[i]],
                                                 max_repeat * max_repeat_block) >= max_repeat:
                outputs[index_map[i]] = decoder_inputs[index_map[i]]
    if is_single:
        outputs = outputs[0]
    return outputs


def baseline_model(AU_count):
    fc_dim = 256
    # create model
    inputs = Input(shape=(224,224,3))

    #block 1
    base_model = InceptionV3(weights="imagenet",include_top=False, input_shape= (224,224,3))
    base_model = Model(inputs=base_model.input,outputs = base_model.get_layer('activation_74').output)
    g = base_model(inputs)

    gh = Conv2D(64, (3,3), padding='same', kernel_initializer='glorot_normal')(g)
    gh1 = Conv2D(AU_count, (1,1), padding='same', kernel_initializer='glorot_normal')(gh)
    gh2 = Conv2D(AU_count, (1,1), padding='same', activation='sigmoid',name = "att_loss", kernel_initializer='glorot_normal')(gh1)
    gh1 = Conv2D(AU_count, (1,1), padding='same', activation='linear', kernel_initializer='glorot_normal')(gh1)
    gap = GlobalAveragePooling2D()(gh1)
    att_output = Activation('sigmoid',name="att_outputs")(gap)
    attention = gh2
    reshape_embed = Reshape([12*12,AU_count])(attention)
    reshape_embed = Permute((2,1))(reshape_embed)

    # AU attention layers
    for i in range(AU_count):

        # layer1 = Lambda(lambda x: tf.expand_dims(attention[...,i],axis=-1))(attention)
        # dimension expansion can be directly done in tf == 2.15
        layer1 = tf.expand_dims(attention[..., i], axis=-1)

        out = Multiply()([layer1,g])
        g = Add()([out,g])
        mt = Conv2D(64, (1,1), padding='same', kernel_initializer='glorot_normal')(g)
        mt = MaxPooling2D(pool_size=7, strides=(1,1),padding = 'same')(mt)
        mt = BatchNormalization()(mt)
        mt = Activation('relu')(mt)
        perception = Flatten()(mt)

        inter = Dense(fc_dim, activation='relu', kernel_initializer='glorot_normal')(perception)

        # tin = Lambda(lambda x: tf.expand_dims(x,axis=1))(inter)
        tin = tf.expand_dims(inter, axis=1)
        if i==0:
            feat_outputs = tin
        else:
            feat_outputs = Concatenate(axis = 1, name = 'feat_outputs_{}'.format(i+1))([feat_outputs,tin])

    feat_outputs_P  = PositionEmbedding(
        input_shape=(None,),
        input_dim = AU_count,
        output_dim = fc_dim,
        mask_zero=0,
        mode=PositionEmbedding.MODE_ADD,)(feat_outputs)
    feat_outputs_P = get_encoders(name = '1',encoder_num=3,
                    input_layer=feat_outputs_P,
                    head_num=8,hidden_dim=fc_dim,
                    dropout_rate=0.1,)

    feat_outputs_P = Flatten()(feat_outputs_P)
    inter = Dense(fc_dim, activation='relu', kernel_initializer='glorot_normal')(feat_outputs_P)
    final = Dense(AU_count,
            activation='sigmoid',
            name = 'per_outputs_{}'.format(AU_count),
            kernel_initializer='glorot_normal')(inter)
    model = Model(inputs=inputs, outputs=[att_output,final,attention,feat_outputs])
    return model

def create_model():
    model=baseline_model(AU_count)
    model.load_weights('./model/fau.h5')
    return model

if __name__ == '__main__':
    model = create_model()