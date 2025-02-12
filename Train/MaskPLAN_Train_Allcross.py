import os
import tensorflow as tf
import numpy as np
import warnings
from MaskPLAN_BaseAll_cross import *
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import os
gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

def parse_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', default='Base', type=str)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--batch', default=32, type=int)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--loops', default=3, type=int) # training data augmented 3 times

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
        enc_layers = 1
        dec_layers = 1

    elif args.model == 'Base':

        embed_dim = 800
        latent_dim = 3072
        num_heads = 16
        enc_layers = 1
        dec_layers = 1

    elif args.model == 'Large':

        embed_dim = 1000
        latent_dim = 4096
        num_heads = 24
        enc_layers = 1
        dec_layers = 1

    class LearningRateLogger(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self._supports_tf_logs = True
    
        def on_epoch_begin(self, epoch, logs=None):
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            print("\nEpoch %03d: Learning rate is %6.8f." % (epoch, lr))

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=2, min_lr=1e-7)
    LR_SCHEDULE = [(0, 1e-5),(150, 5e-6),(200, 2e-6),(250, 1e-6)]

    tf_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

    def lr_schedule(epoch, lr):
        if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
            return lr
        for i in range(len(LR_SCHEDULE)):
            if epoch == LR_SCHEDULE[i][0]:
                return LR_SCHEDULE[i][1]
        return lr
    
    Mask_data = np.load('Processed_data/RPLAN_mask_cross.npz')
    Input_data = np.load('Processed_data/RPLAN_input_cross.npz')
    Output_data = np.load('Processed_data/RPLAN_output_cross.npz')
    frontD = np.concatenate([np.load('Processed_data/RPLAN_frontdoor.npy')]*args.loops)
    bound = np.concatenate([np.load('Processed_data/RPLAN_B.npy')]*args.loops)

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
        
    MaskPLAN = MASKPLAN(embed_dim,latent_dim,num_heads,enc_layers,dec_layers)
    if args.resume == True:
        MaskPLAN.load_weights('MaskPLAN_Trained/All_%s_Single_cross/All' % (args.model))
    checkpoint = tf.keras.callbacks.ModelCheckpoint('MaskPLAN_Trained/All_%s_Single_cross/All' % (args.model),"val_loss",0,True,True,"min","epoch",options=None,initial_value_threshold=None,)
    MaskPLAN.compile(optimizer=optimizer,loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    MaskPLAN.fit([Mask_T,Mask_L,Mask_A,Mask_S,Mask_R,bound,frontD,T_in,L_in,A_in,S_in,R_in],[T_out,A_out,L_out,S_out,R_out],epochs=EPOCHS,batch_size=BATCH_SIZE,validation_split=0.1,callbacks=[reduce_lr,LearningRateLogger(),checkpoint,tf_callback])#CustomLearningRateScheduler(lr_schedule)

if __name__ == "__main__":
    args = parse_args()
    main(args)