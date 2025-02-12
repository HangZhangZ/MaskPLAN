import os
import tensorflow as tf
import numpy as np
import warnings
from MaskPLAN.MaskPLAN_BaseModel import *
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import os
gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

def parse_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', default='Tiny', type=str)
    parser.add_argument('--part', default='Type', type=str)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--batch', default=32, type=int)

    return parser.parse_args()

def main(args):

    np.random.seed(7)
    tf.random.set_seed(7)

    BATCH_SIZE = args.batch
    EPOCHS = args.epoch

    if args.model == 'Tiny':

        embed_dim = 600
        latent_dim = 2048
        num_heads = 8
        enc_layers = 8
        dec_layers = 8

    elif args.model == 'Base':

        embed_dim = 800
        latent_dim = 3072
        num_heads = 12
        enc_layers = 12
        dec_layers = 12

    elif args.model == 'Large':

        embed_dim = 1000
        latent_dim = 4096
        num_heads = 24
        enc_layers = 16
        dec_layers = 16

    class LearningRateLogger(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self._supports_tf_logs = True
    
        def on_epoch_begin(self, epoch, logs=None):
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            print("\nEpoch %03d: Learning rate is %6.8f." % (epoch, lr))

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=1, min_lr=1e-7)
    
    Mask_data = np.load('Processed_data/RPLAN_mask_train.npz')
    Input_data = np.load('Processed_data/RPLAN_input_train.npz')
    Output_data = np.load('Processed_data/RPLAN_output_train.npz')
    frontD = np.load('Processed_data/RPLAN_frontdoor.npy')
    bound = np.load('Processed_data/RPLAN_B.npy')

    file_index = Mask_data.files

    Mask_T = Mask_data[file_index[0]]
    Mask_L = Mask_data[file_index[1]]
    Mask_A = Mask_data[file_index[2]]
    Mask_S = Mask_data[file_index[3]]
    Mask_R = Mask_data[file_index[4]]

    T_in = Input_data[file_index[0]]
    L_in = Input_data[file_index[1]]
    A_in = Input_data[file_index[2]]
    S_in = Input_data[file_index[3]]
    R_in = Input_data[file_index[4]]

    T_out = Output_data[file_index[0]]
    L_out = Output_data[file_index[1]]
    A_out = Output_data[file_index[2]]
    S_out = Output_data[file_index[3]]
    R_out = Output_data[file_index[4]]
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    if args.part == 'Type':
        
        MaskPLAN_T = MaskPLAN_ModelT(embed_dim,latent_dim,num_heads,enc_layers,dec_layers).MaskPLAN_Type
        os.makedirs('MaskPLAN_Trained/Type_%s/Model_T' % (args.model),exist_ok=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint('MaskPLAN_Trained/Type_%s/Model_T' % (args.model),"val_loss",0,True,True,"min","epoch",options=None,initial_value_threshold=None,)
        MaskPLAN_T.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        MaskPLAN_T.fit([Mask_T,Mask_L,Mask_A,Mask_S,Mask_R,bound,frontD,T_in],T_out,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_split=0.1,callbacks=[reduce_lr,LearningRateLogger(),checkpoint])

    elif args.part == 'Location':

        MaskPLAN_L = MaskPLAN_ModelL(embed_dim,latent_dim,num_heads,enc_layers,dec_layers).MaskPLAN_Loc
        os.makedirs('MaskPLAN_Trained/Loc_%s/Model_L' % (args.model),exist_ok=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint('MaskPLAN_Trained/Loc_%s/Model_L' % (args.model),"val_loss",0,True,True,"min","epoch",options=None,initial_value_threshold=None,)
        MaskPLAN_L.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        MaskPLAN_L.fit([Mask_T,Mask_L,Mask_A,Mask_S,Mask_R,bound,frontD,L_in,T_out],L_out,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_split=0.1,callbacks=[reduce_lr,LearningRateLogger(),checkpoint])

    elif args.part == 'Adjacency':

        MaskPLAN_A = MaskPLAN_ModelA(embed_dim,latent_dim,num_heads,enc_layers,dec_layers).MaskPLAN_Ada
        os.makedirs('MaskPLAN_Trained/Ada_%s/Model_A' % (args.model),exist_ok=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint('MaskPLAN_Trained/Ada_%s/Model_A' % (args.model),"val_loss",0,True,True,"min","epoch",options=None,initial_value_threshold=None,)
        MaskPLAN_A.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        MaskPLAN_A.fit([Mask_T,Mask_L,Mask_A,Mask_S,Mask_R,bound,frontD,A_in,T_out,L_out],A_out,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_split=0.1,callbacks=[reduce_lr,LearningRateLogger(),checkpoint])

    elif args.part == 'Area':

        MaskPLAN_S = MaskPLAN_ModelS(embed_dim,latent_dim,num_heads,enc_layers,dec_layers).MaskPLAN_Area
        os.makedirs('MaskPLAN_Trained/Area_%s/Model_A' % (args.model),exist_ok=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint('MaskPLAN_Trained/Area_%s/Model_S' % (args.model),"val_loss",0,True,True,"min","epoch",options=None,initial_value_threshold=None,)
        MaskPLAN_S.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        MaskPLAN_S.fit([Mask_T,Mask_L,Mask_A,Mask_S,Mask_R,bound,frontD,S_in,T_out,L_out,A_out],S_out,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_split=0.1,callbacks=[reduce_lr,LearningRateLogger(),checkpoint])

    elif args.part == 'Region':

        MaskPLAN_R = MaskPLAN_ModelR(embed_dim,latent_dim,num_heads,enc_layers,dec_layers).MaskPLAN_Room
        os.makedirs('MaskPLAN_Trained/Room_%s/Model_R' % (args.model),exist_ok=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint('MaskPLAN_Trained/Room_%s/Model_R' % (args.model),"val_loss",0,True,True,"min","epoch",options=None,initial_value_threshold=None,)
        MaskPLAN_R.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        MaskPLAN_R.fit([Mask_T,Mask_L,Mask_A,Mask_S,Mask_R,bound,frontD,R_in,T_out,L_out,A_out,S_out],R_out,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_split=0.1,callbacks=[reduce_lr,LearningRateLogger(),checkpoint])

if __name__ == "__main__":
    args = parse_args()
    main(args)