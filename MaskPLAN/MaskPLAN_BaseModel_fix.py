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
code = 25
code_dimen = 64

def MaskPLAN_GlobalEncoder(embed_dim,latent_dim,num_heads,enc_layers):
    transformer_units = [latent_dim, embed_dim]
    door_units = [embed_dim, embed_dim / 5]
    bound_units = [embed_dim, embed_dim / 2]
    down_units = [embed_dim / 2, embed_dim / 5]

    type_mask = Input(shape=(sqe_len), dtype="int64", name="type_mask")
    loc_mask = Input(shape=(sqe_len,code), dtype="int64", name="loc_mask")
    ada_mask = Input(shape=(sqe_len,sqe_len-2), dtype="int64", name="ada_inputs")
    area_mask = Input(shape=(sqe_len), dtype="int64", name="area_mask")
    room_mask = Input(shape=(sqe_len,code), dtype="int64", name="room_mask")
    bound_input = Input(shape=(code), dtype="int64", name="bound_input")
    door_input = Input(shape=(2), dtype="int64", name="door_input")

    type_embedding = PositionalEmbedding(sqe_len,type_dimen,embed_dim)(type_mask)
    loc_embedding = PositionalEmbedding_IMG(sqe_len,embed_dim)(loc_mask)
    ada_embedding = PositionalEmbedding_ada(sqe_len,embed_dim)(ada_mask)
    area_embedding = PositionalEmbedding(sqe_len,area_dimen,embed_dim)(area_mask)
    room_embedding = PositionalEmbedding_IMG(sqe_len,embed_dim)(room_mask)
    
    type_embedding = mlp(type_embedding,down_units)
    loc_embedding = mlp(loc_embedding,down_units)
    ada_embedding = mlp(ada_embedding,down_units)
    area_embedding = mlp(area_embedding,down_units)
    room_embedding = mlp(area_embedding,down_units)
    x = Concatenate()([type_embedding,loc_embedding,ada_embedding,area_embedding,room_embedding])

    x = TransformerEncoder(embed_dim, transformer_units, num_heads, enc_layers)(x)

    bound_embedding = Embedding(input_dim=code_dimen, output_dim=embed_dim)(bound_input)
    bound_embedding = Flatten()(bound_embedding)
    bound_embedding = mlp(bound_embedding,bound_units)
    bound_embedding = RepeatVector(sqe_len)(bound_embedding)
    door_embedding = Embedding(input_dim=loc_dimen, output_dim=embed_dim)(door_input)
    door_embedding = Flatten()(door_embedding)
    door_embedding = mlp(door_embedding,door_units)
    door_embedding = RepeatVector(sqe_len)(door_embedding)

    x = Concatenate()([x,bound_embedding,door_embedding])
    encoder_outputs = mlp(x,transformer_units)

    return Model([type_mask,loc_mask,ada_mask,area_mask,room_mask,bound_input,door_input], encoder_outputs, name='MAE_Encoder')

class MaskPLAN_ModelT():
    def __init__(self,embed_dim,latent_dim,num_heads,enc_layers,dec_layers):
        super().__init__()
        transformer_units = [latent_dim, embed_dim]
        type_mask = Input(shape=(sqe_len), dtype="int64")
        loc_mask = Input(shape=(sqe_len,code), dtype="int64")
        ada_mask = Input(shape=(sqe_len,sqe_len-2), dtype="int64")
        area_mask = Input(shape=(sqe_len), dtype="int64")
        room_mask = Input(shape=(sqe_len,code), dtype="int64")
        bound_input = Input(shape=(code), dtype="int64")
        door_input = Input(shape=(2), dtype="int64")

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
        loc_mask = Input(shape=(sqe_len,code), dtype="int64")
        ada_mask = Input(shape=(sqe_len,sqe_len-2), dtype="int64")
        area_mask = Input(shape=(sqe_len), dtype="int64")
        room_mask = Input(shape=(sqe_len,code), dtype="int64")
        bound_input = Input(shape=(code), dtype="int64")
        door_input = Input(shape=(2), dtype="int64")

        MaskPLAN_Encoder = MaskPLAN_GlobalEncoder(embed_dim,latent_dim,num_heads,enc_layers)

        decoder_inputs1 = Input(shape=(sqe_len,code), dtype="int64")
        decoder_inputsE1 = PositionalEmbedding_IMG(sqe_len,embed_dim)(decoder_inputs1)

        encoded_inputs10 = Input(shape=(sqe_len, embed_dim), dtype="float32")
        encoded_inputs11 = Input(shape=(sqe_len),dtype="int64")
        flow_type = PositionalEmbedding(sqe_len,type_dimen,embed_dim)(encoded_inputs11)
        flow_type = Dense((embed_dim), activation="LeakyReLU")(flow_type)

        encoder_outputs1 = Concatenate()([flow_type,encoded_inputs10])
        encoder_outputs1 = mlp(encoder_outputs1,transformer_units)

        x1 = TransformerDecoder(embed_dim, transformer_units, num_heads, dec_layers)(decoder_inputsE1, encoder_outputs1)

        decoder_outputs1 = Dense(code*code_dimen, activation="LeakyReLU")(x1)
        decoder_outputs1 = Reshape((sqe_len, code, code_dimen))(decoder_outputs1)
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
        loc_mask = Input(shape=(sqe_len,code), dtype="int64")
        ada_mask = Input(shape=(sqe_len,sqe_len-2), dtype="int64")
        area_mask = Input(shape=(sqe_len), dtype="int64")
        room_mask = Input(shape=(sqe_len,code), dtype="int64")
        bound_input = Input(shape=(code), dtype="int64")
        door_input = Input(shape=(2), dtype="int64")

        MaskPLAN_Encoder = MaskPLAN_GlobalEncoder(embed_dim,latent_dim,num_heads,enc_layers)

        decoder_inputs2 = Input(shape=(sqe_len,sqe_len-2), dtype="int64")
        decoder_inputsE2 = PositionalEmbedding_ada(sqe_len,embed_dim)(decoder_inputs2)

        encoded_inputs20 = Input(shape=(sqe_len, embed_dim), dtype="float32")
        encoded_inputs21 = Input(shape=(sqe_len),dtype="int64")
        encoded_inputs22 = Input(shape=(sqe_len,code),dtype="int64")

        flow_type = PositionalEmbedding(sqe_len,type_dimen,embed_dim)(encoded_inputs21)
        flow_loc = PositionalEmbedding_IMG(sqe_len,embed_dim)(encoded_inputs22)
        flow_type = layers.Dense((embed_dim / 2), activation="LeakyReLU")(flow_type)
        flow_loc = layers.Dense((embed_dim / 2), activation="LeakyReLU")(flow_loc)

        encoder_outputs2 = Concatenate()([flow_loc,flow_type,encoded_inputs20])
        encoder_outputs2 = mlp(encoder_outputs2,transformer_units)

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
        loc_mask = Input(shape=(sqe_len,code), dtype="int64")
        ada_mask = Input(shape=(sqe_len,sqe_len-2), dtype="int64")
        area_mask = Input(shape=(sqe_len), dtype="int64")
        room_mask = Input(shape=(sqe_len,code), dtype="int64")
        bound_input = Input(shape=(code), dtype="int64")
        door_input = Input(shape=(2), dtype="int64")

        MaskPLAN_Encoder = MaskPLAN_GlobalEncoder(embed_dim,latent_dim,num_heads,enc_layers)

        decoder_inputs3 = Input(shape=(sqe_len), dtype="int64")
        decoder_inputsE3 = PositionalEmbedding(sqe_len,area_dimen,embed_dim)(decoder_inputs3)

        encoded_inputs30 = Input(shape=(sqe_len, embed_dim), dtype="float32")
        encoded_inputs31 = Input(shape=(sqe_len),dtype="int64")
        encoded_inputs32 = Input(shape=(sqe_len,code),dtype="int64")
        encoded_inputs33 = Input(shape=(sqe_len,sqe_len-2),dtype="int64")

        flow_type = PositionalEmbedding(sqe_len,type_dimen,embed_dim)(encoded_inputs31)
        flow_loc = PositionalEmbedding_IMG(sqe_len,embed_dim)(encoded_inputs32)
        flow_ada = PositionalEmbedding_ada(sqe_len,embed_dim)(encoded_inputs33)
        flow_type = layers.Dense((embed_dim / 2), activation="LeakyReLU")(flow_type)
        flow_loc = layers.Dense((embed_dim / 2), activation="LeakyReLU")(flow_loc)
        flow_ada = layers.Dense((embed_dim / 2), activation="LeakyReLU")(flow_ada)

        encoder_outputs3 = Concatenate()([flow_ada,flow_loc,flow_type,encoded_inputs30])
        encoder_outputs3 = mlp(encoder_outputs3,transformer_units)

        x3 = TransformerDecoder(embed_dim, transformer_units, num_heads,dec_layers)(decoder_inputsE3, encoder_outputs3)#, mask1, mask2

        decoder_outputs3 = layers.Dense(area_dimen, activation="LeakyReLU")(x3)
        decoder_outputs3 = layers.Reshape((sqe_len, area_dimen))(decoder_outputs3)
        final_Area = Activation('softmax')(decoder_outputs3)
        Decoder_Area = Model([decoder_inputs3,encoded_inputs31,encoded_inputs32,encoded_inputs33,encoded_inputs30], final_Area, name="Decoder_area")

        get_mask_embedding3 = MaskPLAN_Encoder([type_mask,loc_mask,ada_mask,area_mask,room_mask,bound_input,door_input])
        Final_area = Decoder_Area([decoder_inputs3,encoded_inputs31,encoded_inputs32,encoded_inputs33,get_mask_embedding3])

        self.MaskPLAN_Area = Model([type_mask,loc_mask,ada_mask,area_mask,room_mask,bound_input,door_input,decoder_inputs3,encoded_inputs31,encoded_inputs32,encoded_inputs33], Final_area, name="Mask_area")

class MaskPLAN_ModelR():
    def __init__(self,embed_dim,latent_dim,num_heads,enc_layers,dec_layers):
        super().__init__()
        transformer_units = [latent_dim, embed_dim]
        type_mask = Input(shape=(sqe_len), dtype="int64")
        loc_mask = Input(shape=(sqe_len,code), dtype="int64")
        ada_mask = Input(shape=(sqe_len,sqe_len-2), dtype="int64")
        area_mask = Input(shape=(sqe_len), dtype="int64")
        room_mask = Input(shape=(sqe_len,code), dtype="int64")
        bound_input = Input(shape=(code), dtype="int64")
        door_input = Input(shape=(2), dtype="int64")

        MaskPLAN_Encoder = MaskPLAN_GlobalEncoder(embed_dim,latent_dim,num_heads,enc_layers)

        decoder_inputs4 = Input(shape=(sqe_len,code), dtype="int64")
        decoder_inputsE4 = PositionalEmbedding_IMG(sqe_len,embed_dim)(decoder_inputs4)

        encoded_inputs40 = Input(shape=(sqe_len, embed_dim), dtype="float32")
        encoded_inputs41 = Input(shape=(sqe_len),dtype="int64")
        encoded_inputs42 = Input(shape=(sqe_len,code),dtype="int64")
        encoded_inputs43 = Input(shape=(sqe_len,sqe_len-2),dtype="int64")
        encoded_inputs44 = Input(shape=(sqe_len),dtype="int64")

        flow_type = PositionalEmbedding(sqe_len,type_dimen,embed_dim)(encoded_inputs41)
        flow_loc = PositionalEmbedding_IMG(sqe_len,embed_dim)(encoded_inputs42)
        flow_ada = PositionalEmbedding_ada(sqe_len,embed_dim)(encoded_inputs43)
        flow_area = PositionalEmbedding(sqe_len,area_dimen,embed_dim)(encoded_inputs44)
        flow_type = layers.Dense((embed_dim / 2), activation="LeakyReLU")(flow_type)
        flow_loc = layers.Dense((embed_dim / 2), activation="LeakyReLU")(flow_loc)
        flow_ada = layers.Dense((embed_dim / 2), activation="LeakyReLU")(flow_ada)
        flow_area = layers.Dense((embed_dim / 2), activation="LeakyReLU")(flow_area)

        encoder_outputs4 = Concatenate()([flow_area,flow_ada,flow_loc,flow_type,encoded_inputs40])
        encoder_outputs4 = mlp(encoder_outputs4,transformer_units)

        x4 = TransformerDecoder(embed_dim, transformer_units, num_heads,dec_layers)(decoder_inputsE4, encoder_outputs4)#, mask1, mask2

        decoder_outputs4 = layers.Dense(code*code_dimen, activation="LeakyReLU")(x4)
        decoder_outputs4 = layers.Reshape((sqe_len, code, code_dimen))(decoder_outputs4)
        final_Room = Activation('softmax')(decoder_outputs4)
        Decoder_Room = Model([decoder_inputs4,encoded_inputs41,encoded_inputs42,encoded_inputs43,encoded_inputs44,encoded_inputs40], final_Room, name="Decoder_room")

        get_mask_embedding4 = MaskPLAN_Encoder([type_mask,loc_mask,ada_mask,area_mask,room_mask,bound_input,door_input])
        Final_region = Decoder_Room([decoder_inputs4,encoded_inputs41,encoded_inputs42,encoded_inputs43,encoded_inputs44,get_mask_embedding4])

        self.MaskPLAN_Room = Model([type_mask,loc_mask,ada_mask,area_mask,room_mask,bound_input,door_input,decoder_inputs4,encoded_inputs41,encoded_inputs42,encoded_inputs43,encoded_inputs44], Final_region, name="Mask_room")
