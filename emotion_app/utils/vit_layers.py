"""Keras building blocks for the ViT FER2013 model (best_fer2013_vit.h5).

Imported before ``load_model`` so the custom layers/functions in the saved
h5 can be deserialized (they are registered with ``register_keras_serializable``).
"""
import numpy as np
import keras
from keras import layers, ops

DIM = 768
HEADS = 12
HEAD_DIM = DIM // HEADS
MLP_DIM = 3072
NUM_BLOCKS = 12
NUM_PATCHES = 196
SEQ = 197
EPS = 1e-12


@keras.saving.register_keras_serializable()
class ClsToken(layers.Layer):
    def __init__(self, dim, **kw):
        super().__init__(**kw)
        self.dim = dim

    def build(self, input_shape):
        self.token = self.add_weight(shape=(1, 1, self.dim), initializer="zeros",
                                     trainable=True, name="token")

    def call(self, x):
        return ops.concatenate([ops.broadcast_to(self.token, (ops.shape(x)[0], 1, self.dim)), x], axis=1)

    def get_config(self):
        return {**super().get_config(), "dim": self.dim}


@keras.saving.register_keras_serializable()
class AddPosEmbed(layers.Layer):
    def __init__(self, seq, dim, **kw):
        super().__init__(**kw)
        self.seq = seq
        self.dim = dim

    def build(self, input_shape):
        self.pos = self.add_weight(shape=(1, self.seq, self.dim), initializer="zeros",
                                   trainable=True, name="pos")

    def call(self, x):
        return x + self.pos

    def get_config(self):
        return {**super().get_config(), "seq": self.seq, "dim": self.dim}


@keras.saving.register_keras_serializable()
def take_cls(t):
    return t[:, 0]


@keras.saving.register_keras_serializable()
def qk_scores(args):
    q, kt = args
    return ops.matmul(q, kt) / np.sqrt(HEAD_DIM)


@keras.saving.register_keras_serializable()
def attn_value(args):
    attn, v = args
    return ops.matmul(attn, v)


def mha(x, block):
    y = layers.LayerNormalization(epsilon=EPS, name=f"b{block}_ln1")(x)
    q = layers.Dense(DIM, name=f"b{block}_q")(y)
    k = layers.Dense(DIM, name=f"b{block}_k")(y)
    v = layers.Dense(DIM, name=f"b{block}_v")(y)
    q = layers.Reshape((SEQ, HEADS, HEAD_DIM))(q)
    k = layers.Reshape((SEQ, HEADS, HEAD_DIM))(k)
    v = layers.Reshape((SEQ, HEADS, HEAD_DIM))(v)
    q = layers.Permute((2, 1, 3))(q)   # (B, HEADS, SEQ, HD)
    k = layers.Permute((2, 1, 3))(k)
    v = layers.Permute((2, 1, 3))(v)
    kt = layers.Permute((1, 3, 2))(k)  # (B, HEADS, HD, SEQ)
    scores = layers.Lambda(qk_scores, name=f"b{block}_mm1")([q, kt])
    attn = layers.Softmax(axis=-1)(scores)
    out = layers.Lambda(attn_value, name=f"b{block}_mm2")([attn, v])
    out = layers.Permute((2, 1, 3))(out)
    out = layers.Reshape((SEQ, DIM))(out)
    out = layers.Dense(DIM, name=f"b{block}_attn_out")(out)
    return layers.Add(name=f"b{block}_res1")([x, out])


def mlp(x, block):
    y = layers.LayerNormalization(epsilon=EPS, name=f"b{block}_ln2")(x)
    y = layers.Dense(MLP_DIM, name=f"b{block}_fc1")(y)
    y = layers.Activation(keras.activations.gelu, name=f"b{block}_gelu")(y)
    y = layers.Dense(DIM, name=f"b{block}_fc2")(y)
    return layers.Add(name=f"b{block}_res2")([x, y])


def build_vit():
    """Build the ViT-Base FER2013 model (input [-1,1] 224x224 RGB, output 7 softmax)."""
    inp = layers.Input((224, 224, 3), name="input_1")
    x = layers.Conv2D(DIM, 16, 16, padding="valid", name="patch_proj")(inp)
    x = layers.Reshape((NUM_PATCHES, DIM))(x)
    x = ClsToken(DIM, name="cls_token")(x)
    x = AddPosEmbed(SEQ, DIM, name="add_pos_embed")(x)
    for i in range(NUM_BLOCKS):
        x = mha(x, i)
        x = mlp(x, i)
    x = layers.LayerNormalization(epsilon=EPS, name="final_ln")(x)
    x = layers.Lambda(take_cls, name="cls_take")(x)
    out = layers.Dense(7, activation="softmax", name="classifier")(x)
    return keras.Model(inp, out)