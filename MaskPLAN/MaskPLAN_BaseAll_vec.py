import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Embedding, Flatten, Activation, RepeatVector, Dense, Reshape, Cropping1D, Dropout, BatchNormalization
from .MaskPLAN import *
np.random.seed(42)
tf.random.set_seed(42)

type_dimen = 10
area_dimen = 32
ada_dimen = 2
sqe_len = 10
code = 25
code_dimen = 64

def MASKPLAN(embed_dim,latent_dim,num_heads,enc_layers,dec_layers):
    transformer_units = [latent_dim, embed_dim]
    door_units = embed_dim / 2
    bound_units = embed_dim

    type_mask = Input(shape=(sqe_len), dtype="int64", name="type_mask")
    loc_mask = Input(shape=(sqe_len,2), dtype="int64", name="loc_mask")
    ada_mask = Input(shape=(sqe_len,sqe_len-2), dtype="int64", name="ada_inputs")
    area_mask = Input(shape=(sqe_len), dtype="int64", name="area_mask")
    room_mask = Input(shape=(sqe_len,4), dtype="int64", name="room_mask")
    bound_input = Input(shape=(code), dtype="int64", name="bound_input")
    door_input = Input(shape=(2), dtype="int64", name="door_input")

    type_embedding = PositionalEmbedding(sqe_len,type_dimen,embed_dim)(type_mask)
    loc_embedding = PositionalEmbedding_loc(sqe_len,embed_dim)(loc_mask)
    ada_embedding = PositionalEmbedding_ada(sqe_len,embed_dim)(ada_mask)
    area_embedding = PositionalEmbedding(sqe_len,area_dimen,embed_dim)(area_mask)
    room_embedding = PositionalEmbedding_room(sqe_len,embed_dim)(room_mask)
    
    mask_sqe = Concatenate(axis=1)([type_embedding, ada_embedding, loc_embedding, area_embedding, room_embedding])
    mask_sqe = TransformerEncoder(embed_dim,transformer_units,num_heads,enc_layers)(mask_sqe)
    # mask_sqe = Dropout(0.5)(mask_sqe)

    bound_embedding = Embedding(input_dim=code_dimen, output_dim=embed_dim)(bound_input)
    bound_embedding = Flatten()(bound_embedding)
    bound_embedding = Dense(bound_units,activation='LeakyReLU')(bound_embedding)
    bound_embedding = RepeatVector(sqe_len*5)(bound_embedding)
    door_embedding = Dense(door_units,activation='LeakyReLU')(door_input)
    door_embedding = RepeatVector(sqe_len*5)(door_embedding)

    mask_sqe = Concatenate()([mask_sqe,bound_embedding,door_embedding])
    mask_outputs = mlp(mask_sqe,transformer_units)

    type_in = Input(shape=(sqe_len), dtype="int64", name="type")
    loc_in = Input(shape=(sqe_len,2), dtype="int64", name="loc")
    ada_in = Input(shape=(sqe_len,sqe_len-2), dtype="int64", name="ada")
    area_in = Input(shape=(sqe_len), dtype="int64", name="area")
    room_in = Input(shape=(sqe_len,4), dtype="int64", name="room")

    type_embedding2 = PositionalEmbedding(sqe_len,type_dimen,embed_dim)(type_in)
    loc_embedding2 = PositionalEmbedding_loc(sqe_len,embed_dim)(loc_in)
    ada_embedding2 = PositionalEmbedding_ada(sqe_len,embed_dim)(ada_in)
    area_embedding2 = PositionalEmbedding(sqe_len,area_dimen,embed_dim)(area_in)
    room_embedding2 = PositionalEmbedding_room(sqe_len,embed_dim)(room_in)

    in_sqe = Concatenate(axis=1)([type_embedding2, ada_embedding2, loc_embedding2, area_embedding2, room_embedding2])

    Generator_Encoder = GeneratorEncoder(embed_dim,transformer_units,num_heads,enc_layers)
    x = Transformer_Encoder_Decoder(embed_dim, transformer_units, num_heads, dec_layers)(in_sqe, mask_outputs)
    # x = Dropout(0.5)(x)

    mask_outputs = Generator_Encoder(mask_outputs,x)

    x0 = Cropping1D(cropping=(0,40))(x)
    x0 = mlp(x0,[latent_dim,embed_dim,type_dimen])
    x0 = Reshape((sqe_len, type_dimen))(x0)
    x0 = BatchNormalization()(x0)
    x0 = Activation('softmax',name='T')(x0)

    x1 = Cropping1D(cropping=(10,30))(x)
    x1 = mlp(x1,[latent_dim,embed_dim,int((sqe_len-2)*2)])
    x1 = Reshape((sqe_len, sqe_len-2,2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('softmax',name='A')(x1)

    x2 = Cropping1D(cropping=(20,20))(x)
    x2 = mlp(x2,[latent_dim,embed_dim,loc_dimen*2])
    x2 = Reshape((sqe_len, 2, loc_dimen))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('softmax',name='L')(x2)

    x3 = Cropping1D(cropping=(30,10))(x)
    x3 = mlp(x3,[latent_dim,embed_dim,area_dimen])
    x3 = Reshape((sqe_len, area_dimen))(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('softmax',name='S')(x3)

    x4 = Cropping1D(cropping=(40,0))(x)
    x4 = mlp(x4,[latent_dim,embed_dim,room_dimen*4])
    x4 = Reshape((sqe_len, 4, room_dimen))(x4)
    x4 = BatchNormalization()(x4)
    x4 = Activation('softmax',name='R')(x4)

    return Model([type_mask,loc_mask,ada_mask,area_mask,room_mask,bound_input,door_input,type_in,loc_in,ada_in,area_in,room_in],[x0,x1,x2,x3,x4])

