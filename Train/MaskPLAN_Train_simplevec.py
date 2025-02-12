import os
import tensorflow as tf
import numpy as np
import warnings
from MaskPLAN_BaseModel_simplevec import *
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import os
gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
np.random.seed(42)
tf.random.set_seed(42)
import datetime

def parse_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', default='Base', type=str)
    parser.add_argument('--part', default='Type', type=str)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--batch', default=32, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--val', default=1e-1, type=float)
    parser.add_argument('--lrmethod', default='stage', type=str)

    return parser.parse_args()

def main(args):

    BATCH_SIZE = args.batch
    EPOCHS = args.epoch

    if args.model == 'Tiny':

        embed_dim = 600
        latent_dim = 2048
        num_heads = 8
        enc_layers = 6
        dec_layers = 8

    elif args.model == 'Base':

        embed_dim = 800
        latent_dim = 3072
        num_heads = 12
        enc_layers = 1
        dec_layers = 1

    elif args.model == 'Large':

        embed_dim = 1000
        latent_dim = 4096
        num_heads = 24
        enc_layers = 1
        dec_layers = 1

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=2, min_lr=1e-7)
    
    LR_SCHEDULE = [(0, 1e-5),(25, 5e-6),(50, 2e-6),(75, 1e-6)]

    def lr_schedule(epoch, lr):
        if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
            return lr
        for i in range(len(LR_SCHEDULE)):
            if epoch == LR_SCHEDULE[i][0]:
                return LR_SCHEDULE[i][1]
        return lr

    Mask_data = np.load('Processed_data/RPLAN_mask_train.npz')
    Input_data = np.load('Processed_data/RPLAN_input_train.npz')
    Output_data = np.load('Processed_data/RPLAN_output_train.npz')
    frontD = np.load('Processed_data/RPLAN_frontdoor.npy')
    bound = np.load('Processed_data/RPLAN_bound.npy',allow_pickle=True).reshape(80788,64)/255.

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

    # log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # wandb.init(sync_tensorboard=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('MaskPLAN_Trained/simple_%s_%s_vec/Model' % (args.part,args.model),"loss",0,True,True,"min","epoch",options=None,initial_value_threshold=None,)

    if args.lrmethod == 'stage':
        callback = [CustomLearningRateScheduler(lr_schedule),checkpoint]
    elif args.lrmethod == 'checkval':
        callback = [reduce_lr,LearningRateLogger(),checkpoint]

    if args.part == 'Type':
        
        MaskPLAN_T = MaskPLAN_ModelT(embed_dim,latent_dim,num_heads,enc_layers,dec_layers).MaskPLAN_Type
        MaskPLAN_T.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        MaskPLAN_T.fit([Mask_T,Mask_L,Mask_A,Mask_S,Mask_R,bound,frontD,T_in],T_out,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_split=args.val,shuffle=True,callbacks=callback)

    elif args.part == 'Location':

        MaskPLAN_L = MaskPLAN_ModelL(embed_dim,latent_dim,num_heads,enc_layers,dec_layers).MaskPLAN_Loc
        # if args.resume == True:
        #     MaskPLAN_L.load_weights('MaskPLAN_Trained/Loc_%s_vec/Model_L' % (args.model))
        MaskPLAN_L.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        MaskPLAN_L.fit([Mask_T,Mask_L,Mask_A,Mask_S,Mask_R,bound,frontD,L_in,T_out],L_out,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_split=args.val,shuffle=True,callbacks=callback)

    elif args.part == 'Adjacency':

        MaskPLAN_A = MaskPLAN_ModelA(embed_dim,latent_dim,num_heads,enc_layers,dec_layers).MaskPLAN_Ada
        MaskPLAN_A.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        MaskPLAN_A.fit([Mask_T,Mask_L,Mask_A,Mask_S,Mask_R,bound,frontD,A_in,T_out,L_out],A_out,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_split=args.val,shuffle=True,callbacks=callback)

    elif args.part == 'Area':

        MaskPLAN_S = MaskPLAN_ModelS(embed_dim,latent_dim,num_heads,enc_layers,dec_layers).MaskPLAN_Area
        MaskPLAN_S.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        MaskPLAN_S.fit([T_out,L_out,A_out,Mask_S,Mask_R,bound,frontD,S_in],S_out,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_split=args.val,shuffle=True,callbacks=callback)

    elif args.part == 'Region':

        MaskPLAN_R = MaskPLAN_ModelR(embed_dim,latent_dim,num_heads,enc_layers,dec_layers).MaskPLAN_Room
        MaskPLAN_R.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        MaskPLAN_R.fit([Mask_T,Mask_L,Mask_A,Mask_S,Mask_R,bound,frontD,R_in,T_out,L_out,A_out,S_out],R_out,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_split=args.val,shuffle=True,callbacks=callback)

if __name__ == "__main__":
    args = parse_args()
    main(args)