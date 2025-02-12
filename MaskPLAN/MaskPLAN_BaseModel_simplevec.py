import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Embedding, Flatten, Activation, RepeatVector, Dense, Reshape 
# from MaskPLAN_Modules import *
from .MaskPLAN import *
np.random.seed(42)
tf.random.set_seed(42)

type_dimen = 10
area_dimen = 32
ada_dimen = 2
sqe_len = 10
# code = 25
code_dimen = 64

def MaskPLAN_GlobalEncoder(embed_dim,latent_dim,num_heads,enc_layers):
    transformer_units = [latent_dim, embed_dim]

    type_mask = Input(shape=(sqe_len), dtype="int64", name="type_mask")
    loc_mask = Input(shape=(sqe_len,2), dtype="int64", name="loc_mask")
    ada_mask = Input(shape=(sqe_len,sqe_len-2), dtype="int64", name="ada_inputs")
    area_mask = Input(shape=(sqe_len), dtype="int64", name="area_mask")
    room_mask = Input(shape=(sqe_len,4), dtype="int64", name="room_mask")
    bound_input = Input(shape=(code_dimen), dtype="float32", name="bound_input")
    door_input = Input(shape=(2), dtype="float32", name="door_input")

    type_embedding = PositionalEmbedding(sqe_len,type_dimen,embed_dim)(type_mask)
    loc_embedding = PositionalEmbedding_loc(sqe_len,embed_dim)(loc_mask)
    ada_embedding = PositionalEmbedding_ada(sqe_len,embed_dim)(ada_mask)
    area_embedding = PositionalEmbedding(sqe_len,area_dimen,embed_dim)(area_mask)
    room_embedding = PositionalEmbedding_room(sqe_len,embed_dim)(room_mask)
    
    feature_embedding = Concatenate()([type_embedding,loc_embedding,ada_embedding,area_embedding,room_embedding])
    feature_embedding = mlp(feature_embedding,transformer_units)

    x = TransformerEncoder(embed_dim, transformer_units, num_heads, enc_layers)(feature_embedding)

    bound_embedding = Dense((embed_dim/2), activation="LeakyReLU")(bound_input)
    bound_embedding = RepeatVector(10)(bound_embedding)

    door_embedding = Dense((embed_dim/4), activation="LeakyReLU")(door_input)
    door_embedding = RepeatVector(10)(door_embedding)
    x = Concatenate()([x,bound_embedding,door_embedding])
    encoder_outputs = Dense((embed_dim), activation="LeakyReLU")(x)

    return Model([type_mask,loc_mask,ada_mask,area_mask,room_mask,bound_input,door_input], encoder_outputs, name='MAE_Encoder')

class MaskPLAN_ModelT():
    def __init__(self,embed_dim,latent_dim,num_heads,enc_layers,dec_layers):
        super().__init__()
        transformer_units = [latent_dim, embed_dim]
        type_mask = Input(shape=(sqe_len), dtype="int64")
        loc_mask = Input(shape=(sqe_len,2), dtype="int64")
        ada_mask = Input(shape=(sqe_len,sqe_len-2), dtype="int64")
        area_mask = Input(shape=(sqe_len), dtype="int64")
        room_mask = Input(shape=(sqe_len,4), dtype="int64")
        bound_input = Input(shape=(code_dimen), dtype="float32")
        door_input = Input(shape=(2), dtype="float32")

        MaskPLAN_Encoder = MaskPLAN_GlobalEncoder(embed_dim,latent_dim,num_heads,enc_layers)

        decoder_inputs0 = Input(shape=(sqe_len), dtype="int64")
        decoder_inputsE0 = PositionalEmbedding(sqe_len,type_dimen,embed_dim)(decoder_inputs0)
        encoded_inputs0 = Input(shape=(sqe_len,embed_dim), dtype="float32")

        x0 = TransformerDecoder(embed_dim, transformer_units, num_heads, dec_layers)(decoder_inputsE0, encoded_inputs0)

        decoder_outputs0 = Dense(type_dimen, activation="LeakyReLU")(x0)
        decoder_outputs0 = Reshape((sqe_len, type_dimen))(decoder_outputs0)
        decoder_outputs0 = Activation('softmax')(decoder_outputs0)

        Decoder_Type = Model([decoder_inputs0,encoded_inputs0],decoder_outputs0,name='Decoder_Type')

        get_mask_embedding0 = MaskPLAN_Encoder([type_mask,loc_mask,ada_mask,area_mask,room_mask,bound_input,door_input])
        final_type = Decoder_Type([decoder_inputs0,get_mask_embedding0])  
        
        self.MaskPLAN_Type = Model([type_mask,loc_mask,ada_mask,area_mask,room_mask,bound_input,door_input,decoder_inputs0], final_type, name='Mask_Type')

class MaskPLAN_ModelL():
    def __init__(self,embed_dim,latent_dim,num_heads,enc_layers,dec_layers):
        super().__init__()
        transformer_units = [latent_dim, embed_dim]
        type_mask = Input(shape=(sqe_len), dtype="int64")
        loc_mask = Input(shape=(sqe_len,2), dtype="int64")
        ada_mask = Input(shape=(sqe_len,sqe_len-2), dtype="int64")
        area_mask = Input(shape=(sqe_len), dtype="int64")
        room_mask = Input(shape=(sqe_len,4), dtype="int64")
        bound_input = Input(shape=(code_dimen), dtype="float32")
        door_input = Input(shape=(2), dtype="float32")
        enc_layers2 = int(enc_layers/2)

        MaskPLAN_Encoder = MaskPLAN_GlobalEncoder(embed_dim,latent_dim,num_heads,enc_layers)

        decoder_inputs1 = Input(shape=(sqe_len,2), dtype="int64")
        decoder_inputsE1 = PositionalEmbedding_loc(sqe_len,embed_dim)(decoder_inputs1)

        encoded_inputs10 = Input(shape=(sqe_len, embed_dim), dtype="float32")
        encoded_inputs11 = Input(shape=(sqe_len),dtype="int64")
        flow_type = PositionalEmbedding(sqe_len,type_dimen,embed_dim)(encoded_inputs11)

        Generator_Encoder = GeneratorEncoder(embed_dim,transformer_units,num_heads,enc_layers2)
        encoder_outputs1 = Generator_Encoder(flow_type,encoded_inputs10)
        encoder_outputs1 = mlp(Concatenate()([encoder_outputs1,encoded_inputs10]),transformer_units)

        x1 = TransformerDecoder(embed_dim, transformer_units, num_heads, dec_layers)(decoder_inputsE1, encoder_outputs1)

        decoder_outputs1 = Dense(2*loc_dimen, activation="LeakyReLU")(x1)
        decoder_outputs1 = Reshape((sqe_len, 2, loc_dimen))(decoder_outputs1)
        final_Loc = Activation('softmax')(decoder_outputs1)
        Decoder_Loc = Model([decoder_inputs1,encoded_inputs11,encoded_inputs10], final_Loc, name="Decoder_loc")

        get_mask_embedding1 = MaskPLAN_Encoder([type_mask,loc_mask,ada_mask,area_mask,room_mask,bound_input,door_input])
        Final_loc = Decoder_Loc([decoder_inputs1,encoded_inputs11,get_mask_embedding1])

        self.MaskPLAN_Loc = Model([type_mask,loc_mask,ada_mask,area_mask,room_mask,bound_input,door_input,decoder_inputs1,encoded_inputs11], Final_loc, name="Mask_loc")

class MaskPLAN_ModelA():
    def __init__(self,embed_dim,latent_dim,num_heads,enc_layers,dec_layers):
        super().__init__()
        transformer_units = [latent_dim, embed_dim]
        type_mask = Input(shape=(sqe_len), dtype="int64")
        loc_mask = Input(shape=(sqe_len,2), dtype="int64")
        ada_mask = Input(shape=(sqe_len,sqe_len-2), dtype="int64")
        area_mask = Input(shape=(sqe_len), dtype="int64")
        room_mask = Input(shape=(sqe_len,4), dtype="int64")
        bound_input = Input(shape=(code_dimen), dtype="float32")
        door_input = Input(shape=(2), dtype="float32")
        enc_layers2 = int(enc_layers/2)

        MaskPLAN_Encoder = MaskPLAN_GlobalEncoder(embed_dim,latent_dim,num_heads,enc_layers)

        decoder_inputs2 = Input(shape=(sqe_len,sqe_len-2), dtype="int64")
        decoder_inputsE2 = PositionalEmbedding_ada(sqe_len,embed_dim)(decoder_inputs2)

        encoded_inputs20 = Input(shape=(sqe_len, embed_dim), dtype="float32")
        encoded_inputs21 = Input(shape=(sqe_len),dtype="int64")
        encoded_inputs22 = Input(shape=(sqe_len,2),dtype="int64")

        flow_type = PositionalEmbedding(sqe_len,type_dimen,embed_dim)(encoded_inputs21)
        flow_loc = PositionalEmbedding_loc(sqe_len,embed_dim)(encoded_inputs22)

        Generator_Encoder = GeneratorEncoder(embed_dim,transformer_units,num_heads,enc_layers2)
        encoder_outputs2 = Generator_Encoder(Dense(embed_dim, activation="LeakyReLU")(Concatenate()([flow_type,flow_loc])),encoded_inputs20)
        encoder_outputs2 = mlp(Concatenate()([encoder_outputs2,encoded_inputs20]),transformer_units)

        x2 = TransformerDecoder(embed_dim, transformer_units, num_heads,dec_layers)(decoder_inputsE2, encoder_outputs2)#, mask1, mask2

        decoder_outputs2 = layers.Dense(int((sqe_len-2)*2), activation="LeakyReLU")(x2)
        decoder_outputs2 = layers.Reshape((sqe_len, sqe_len-2,2))(decoder_outputs2)
        final_Ada = Activation('softmax')(decoder_outputs2)
        Decoder_Ada = Model([decoder_inputs2,encoded_inputs21,encoded_inputs22,encoded_inputs20], final_Ada, name="Decoder_ada")

        get_mask_embedding2 = MaskPLAN_Encoder([type_mask,loc_mask,ada_mask,area_mask,room_mask,bound_input,door_input])
        Final_ada = Decoder_Ada([decoder_inputs2,encoded_inputs21,encoded_inputs22,get_mask_embedding2])

        self.MaskPLAN_Ada = Model([type_mask,loc_mask,ada_mask,area_mask,room_mask,bound_input,door_input,decoder_inputs2,encoded_inputs21,encoded_inputs22], Final_ada, name="Mask_ada")

class MaskPLAN_ModelS():
    def __init__(self,embed_dim,latent_dim,num_heads,enc_layers,dec_layers):
        super().__init__()
        transformer_units = [latent_dim, embed_dim]
        type_mask = Input(shape=(sqe_len), dtype="int64")
        loc_mask = Input(shape=(sqe_len,2), dtype="int64")
        ada_mask = Input(shape=(sqe_len,sqe_len-2), dtype="int64")
        area_mask = Input(shape=(sqe_len), dtype="int64")
        room_mask = Input(shape=(sqe_len,4), dtype="int64")
        bound_input = Input(shape=(code_dimen), dtype="float32")
        door_input = Input(shape=(2), dtype="float32")

        MaskPLAN_Encoder = MaskPLAN_GlobalEncoder(embed_dim,latent_dim,num_heads,enc_layers)

        decoder_inputs3 = Input(shape=(sqe_len), dtype="int64")
        decoder_inputsE3 = PositionalEmbedding(sqe_len,area_dimen,embed_dim)(decoder_inputs3)

        encoded_inputs30 = Input(shape=(sqe_len, embed_dim), dtype="float32")
        # encoded_inputs31 = Input(shape=(sqe_len),dtype="int64")
        # encoded_inputs32 = Input(shape=(sqe_len,2),dtype="int64")
        # encoded_inputs33 = Input(shape=(sqe_len,sqe_len-2),dtype="int64")

        # flow_type = PositionalEmbedding(sqe_len,type_dimen,embed_dim)(encoded_inputs31)
        # flow_loc = PositionalEmbedding_loc(sqe_len,embed_dim)(encoded_inputs32)
        # flow_ada = PositionalEmbedding_ada(sqe_len,embed_dim)(encoded_inputs33)

        # encoder_outputs3 = mlp((Concatenate()([flow_type,flow_loc,flow_ada,encoded_inputs30])),transformer_units)

        x3 = TransformerDecoder(embed_dim, transformer_units, num_heads,dec_layers)(decoder_inputsE3, encoded_inputs30)#, mask1, mask2

        decoder_outputs3 = layers.Dense(area_dimen, activation="LeakyReLU")(x3)
        decoder_outputs3 = layers.Reshape((sqe_len, area_dimen))(decoder_outputs3)
        final_Area = Activation('softmax')(decoder_outputs3)
        Decoder_Area = Model([decoder_inputs3,encoded_inputs30], final_Area, name="Decoder_area")

        get_mask_embedding3 = MaskPLAN_Encoder([type_mask,loc_mask,ada_mask,area_mask,room_mask,bound_input,door_input])
        Final_area = Decoder_Area([decoder_inputs3,get_mask_embedding3])

        self.MaskPLAN_Area = Model([type_mask,loc_mask,ada_mask,area_mask,room_mask,bound_input,door_input,decoder_inputs3], Final_area, name="Mask_area")

class MaskPLAN_ModelR():
    def __init__(self,embed_dim,latent_dim,num_heads,enc_layers,dec_layers):
        super().__init__()
        transformer_units = [latent_dim, embed_dim]
        type_mask = Input(shape=(sqe_len), dtype="int64")
        loc_mask = Input(shape=(sqe_len,2), dtype="int64")
        ada_mask = Input(shape=(sqe_len,sqe_len-2), dtype="int64")
        area_mask = Input(shape=(sqe_len), dtype="int64")
        room_mask = Input(shape=(sqe_len,4), dtype="int64")
        bound_input = Input(shape=(code_dimen), dtype="float32")
        door_input = Input(shape=(2), dtype="float32")
        enc_layers2 = int(enc_layers/2)
        prior_weight = 2

        MaskPLAN_Encoder = MaskPLAN_GlobalEncoder(embed_dim,latent_dim,num_heads,enc_layers)

        decoder_inputs4 = Input(shape=(sqe_len,4), dtype="int64")
        decoder_inputsE4 = PositionalEmbedding_room(sqe_len,embed_dim)(decoder_inputs4)

        encoded_inputs40 = Input(shape=(sqe_len, embed_dim), dtype="float32")
        encoded_inputs41 = Input(shape=(sqe_len),dtype="int64")
        encoded_inputs42 = Input(shape=(sqe_len,2),dtype="int64")
        encoded_inputs43 = Input(shape=(sqe_len,sqe_len-2),dtype="int64")
        encoded_inputs44 = Input(shape=(sqe_len),dtype="int64")

        flow_type = PositionalEmbedding(sqe_len,type_dimen,embed_dim)(encoded_inputs41)
        flow_loc = PositionalEmbedding_loc(sqe_len,embed_dim)(encoded_inputs42)
        flow_ada = PositionalEmbedding_ada(sqe_len,embed_dim)(encoded_inputs43)
        flow_area = PositionalEmbedding(sqe_len,area_dimen,embed_dim)(encoded_inputs44)

        Generator_Encoder = GeneratorEncoder(embed_dim,transformer_units,num_heads,enc_layers2)
        encoder_outputs4 = Generator_Encoder(Dense(embed_dim, activation="LeakyReLU")(Concatenate()([flow_type,flow_loc,flow_ada,flow_area])),encoded_inputs40)
        encoder_outputs4 = mlp(Concatenate()([Dense(prior_weight*embed_dim)(encoder_outputs4),encoded_inputs40]),transformer_units)

        x4 = TransformerDecoder(embed_dim, transformer_units, num_heads,dec_layers)(decoder_inputsE4, encoder_outputs4)#, mask1, mask2

        decoder_outputs4 = layers.Dense(4*room_dimen, activation="LeakyReLU")(x4)
        decoder_outputs4 = layers.Reshape((sqe_len, 4, room_dimen))(decoder_outputs4)
        final_Room = Activation('softmax')(decoder_outputs4)
        Decoder_Room = Model([decoder_inputs4,encoded_inputs41,encoded_inputs42,encoded_inputs43,encoded_inputs44,encoded_inputs40], final_Room, name="Decoder_room")

        get_mask_embedding4 = MaskPLAN_Encoder([type_mask,loc_mask,ada_mask,area_mask,room_mask,bound_input,door_input])
        Final_region = Decoder_Room([decoder_inputs4,encoded_inputs41,encoded_inputs42,encoded_inputs43,encoded_inputs44,get_mask_embedding4])

        self.MaskPLAN_Room = Model([type_mask,loc_mask,ada_mask,area_mask,room_mask,bound_input,door_input,decoder_inputs4,encoded_inputs41,encoded_inputs42,encoded_inputs43,encoded_inputs44], Final_region, name="Mask_room")
