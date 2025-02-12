import os
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import numpy as np
from tensorflow.keras.models import load_model
import json
import cv2
from Inference.decode_function import *
from utils import mask_ada
from tensorflow.keras.preprocessing import image
import argparse
from shapely.geometry import Polygon,MultiPolygon,Point,LinearRing,MultiPoint,LineString
from shapely.ops import unary_union
from shapely import affinity

## Global Para ##

list_len = 10
type_dimen = 10
loc_dimen = 20
ada_dimen = 2
area_dimen = 32
room_dimen = 20
sqe_len = 10
code = 25
code_dimen = 64
room_order = [2,3,5,6,4,1,7] # order when implement post-process

# color tag for each room type, just for visualization
T_list = [[255,255,255,255],[255,255,0,255],[255,0,255,255],[0,255,255,255],
          [0,0,255,255],[255,0,0,255],[0,255,0,255],[127,127,255,255],[127,255,127,255]]

## Load Data ##

frontD = np.load('Processed_data/RPLAN_frontdoor.npy') # vec
bound = np.load('Processed_data/RPLAN_B.npy') # visual tokens
Testset_ids = np.load('Processed_data/Test_set.npy') # ids
bound_domain = np.load('Processed_data/Processed_data/Boundary_BoundingBox.npy') # vec

## args ##

def parse_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', default='Base', type=str)
    parser.add_argument('--deep', default='Deep', type=str,
                        help='model with single or multi layers, Deep -> multi, Single -> single')
    parser.add_argument('--test_cases', default=1000, type=int) # test layouts
    parser.add_argument('--par_T', default=0.25, type=float) # partial input Type
    parser.add_argument('--par_L', default=0.25, type=float) # partial input Location
    parser.add_argument('--par_A', default=0.25, type=float) # partial input Adjacency 
    parser.add_argument('--par_S', default=0.25, type=float) # partial input Size 
    parser.add_argument('--par_R', default=0.25, type=float) # partial input Region 
    parser.add_argument('--format', default='cross', type=str, 
                        help='cross -> vec+img data, vec -> only vec data, hybrid -> only R to be img')
    parser.add_argument('--skip_idx', default=1, type=int,
                        help='which token to skip, default skip first [start] token')
    
    return parser.parse_args()

## main inference ##

def main(args):

    skip_idx = args.skip_idx
    os.makedirs('Inference/%s_%s_%s/iteration/raw' % (args.model,args.deep,args.format))
    os.makedirs('Inference/%s_%s_%s/iteration/post' % (args.model,args.deep,args.format))

    ## define model

    if args.format == 'cross':

        dimen_L = code
        dimen_R = code

        if args.deep == 'Deep': from MaskPLAN.MaskPLAN_BaseAll_crossDeep import *
        elif args.deep == 'Single': from MaskPLAN.MaskPLAN_BaseAll_cross import *
    
    elif args.format == 'vec':

        dimen_L = 2
        dimen_R = 4

        if args.deep == 'Deep': from MaskPLAN.MaskPLAN_BaseAll_vecDeep import *
        elif args.deep == 'Single': from MaskPLAN.MaskPLAN_BaseAll_vec import *    

    if args.model == 'Tiny':

        embed_dim = 600
        latent_dim = 2048
        num_heads = 8
        enc_layers = 4
        dec_layers = 4

    elif args.model == 'Base':

        embed_dim = 800
        latent_dim = 3072
        num_heads = 16
        enc_layers = 4
        dec_layers = 6

    elif args.model == 'Large':

        embed_dim = 1000
        latent_dim = 4096
        num_heads = 24
        enc_layers = 8
        dec_layers = 8
    
    # overide layer depth if deep -> single
    if args.deep == 'Single':

        enc_layers = 1
        dec_layers = 1

    # load MaskPLAN

    MaskPLAN = MASKPLAN(embed_dim,latent_dim,num_heads,enc_layers,dec_layers)
    MaskPLAN.load_weights('MaskPLAN_Trained/All_%s_%s_%s/All' % (args.model,args.deep,args.format))

    # load VQVAE (shared with Location and Region) 
        
    if args.format != 'vec':

        VQ_encoder = load_model('VQ_Pretrained/mix_5564/vqvae_de/decoder.keras')
        VQ_decoder = load_model('VQ_Pretrained/mix_5564/vqvae_de/decoder.keras')
        VQ_value = np.load('Processed_data/VQ_Pretrained/mix_5564/vqvae_q/quantizer.npy')
        quantizer = VectorQuantizer(code_dimen, code_dimen)
        quantizer.embeddings = tf.Variable(initial_value=VQ_value,trainable=False,name="embeddings_vqvae_R",)

    # load data
    
    Input_data = np.load('Processed_data/RPLAN_input_%s.npz' % (args.format))
    file_index = Input_data.files

    T_in = Input_data[file_index[0]]
    L_in = Input_data[file_index[1]]
    A_in = Input_data[file_index[2]]
    S_in = Input_data[file_index[3]]
    R_in = Input_data[file_index[4]]

    class MaskPLAN_Inference():
            
        def __init__(self):
            super().__init__()

            self.M_T = np.zeros((1,list_len))
            self.M_L = np.zeros((1,list_len,dimen_L))
            self.M_A = np.zeros((1,list_len,list_len-2))
            self.M_S = np.zeros((1,list_len))
            self.M_R = np.zeros((1,list_len,dimen_R))

            self.In_T = np.zeros((1,list_len))
            self.In_L = np.zeros((1,list_len,dimen_L))
            self.In_A = np.zeros((1,list_len,list_len-2))
            self.In_S = np.zeros((1,list_len))
            self.In_R = np.zeros((1,list_len,dimen_R))

        def reset(self,site_id):

            self.bound = bound[site_id].reshape((1,25))
            self.fd = frontD[site_id] .reshape((1,2))

            self.M_T[0,:skip_idx] = T_in[site_id,:skip_idx]
            self.M_L[0,:skip_idx,:] = L_in[site_id,:skip_idx,:]
            self.M_A[0,:skip_idx,:] = A_in[site_id,:skip_idx,:]
            self.M_S[0,:skip_idx] = S_in[site_id,:skip_idx]
            self.M_R[0,:skip_idx,:] = R_in[site_id,:skip_idx,:]

            self.In_T[0,:skip_idx] = T_in[site_id,:skip_idx]
            self.In_L[0,:skip_idx,:] = L_in[site_id,:skip_idx,:]
            self.In_A[0,:skip_idx,:] = A_in[site_id,:skip_idx,:]
            self.In_S[0,:skip_idx] = S_in[site_id,:skip_idx]
            self.In_R[0,:skip_idx,:] = R_in[site_id,:skip_idx,:]

            self.M_T[0,skip_idx:] = 0
            self.M_L[0,skip_idx:] = 0
            self.M_A[0,skip_idx:] = 0
            self.M_S[0,skip_idx:] = 0
            self.M_R[0,skip_idx:] = 0

            self.In_T[0,skip_idx:] = 0
            self.In_L[0,skip_idx:] = 0
            self.In_A[0,skip_idx:] = 0
            self.In_S[0,skip_idx:] = 0
            self.In_R[0,skip_idx:] = 0

        def partial_input(self,site_id):

            valid = (T_in[site_id]==type_dimen-2).argmax(axis=0)-1

            partial_T = np.random.choice(valid-1, round(valid*args.par_T), replace=False) + 1
            self.M_T[0,partial_T] = T_in[site_id,partial_T]
            self.In_T[0,partial_T] = T_in[site_id,partial_T]

            partial_L = np.random.choice(valid-1, round(valid*args.par_L), replace=False) + 1
            self.M_L[0,partial_L] = L_in[site_id,partial_L]
            self.In_L[0,partial_L] = L_in[site_id,partial_L]

            partial_S = np.random.choice(valid-1, round(valid*args.par_S), replace=False) + 1
            self.M_S[0,partial_S] = S_in[site_id,partial_S]
            self.In_S[0,partial_S] = S_in[site_id,partial_S]

            partial_R = np.random.choice(valid-1, round(valid*args.par_R), replace=False) + 1
            self.M_R[0,partial_R] = R_in[site_id,partial_R]
            self.In_R[0,partial_R] = R_in[site_id,partial_R]

            partial_A = mask_ada(np.random.choice(valid, round(valid*args.par_A), replace=False),list_len,A_in[site_id])
            self.M_A[0] = partial_A
            self.In_A[0] = partial_A

        def inference_interation(self,site_id):

            # get boudnary

            pts = []
            types = []
            unioned_rooms = None

            boundary = cv2.imread('parsed_img/img_room_sqe/0/%d.png' % (site_id),cv2.IMREAD_UNCHANGED)[:,:,0]
            boundary[np.where(boundary>100)] = 255
            boundary_pt = get_bound_pt(boundary)
            if boundary_pt.ndim == 1:
                boundary_pt = [[0,0],[0,127],[127,127],[127,0]]
            boundary_line = LinearRing(boundary_pt)
            boundary_outside = Polygon([[0,0],[0,127],[127,127],[127,0]]).difference(Polygon(boundary_line))

            # fill boundary in img

            reconstructed = np.zeros((128,128,4))
            cv2.fillPoly(reconstructed, [np.array(boundary_line.buffer(2).exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], color=255)
            cv2.fillPoly(reconstructed, [np.array(boundary_line.exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], color=0)
            reconst_post = reconstructed.copy()
        
            # Inference, skip all [start] token and defined token
        
            for k in range(int(list_len*5-1)):

                if 0<= k < list_len-1: # Type

                    if self.In_T[0,k+1] != 0: continue
                    else: 
                        pred = MaskPLAN.predict([self.M_T,self.M_L,self.M_A,self.M_S,self.M_R,self.bound,self.fd,self.In_T,self.In_L,self.In_A,self.In_S,self.In_R])[0,0]
                        self.In_T[0,k+1] = np.array(np.argmax(pred[k, :]))
                        if self.In_T[0,k+1] == type_dimen-2: num_room = k

                elif list_len-1+1<= k < list_len*2-1: # Ada

                    pred = MaskPLAN.predict([self.M_T,self.M_L,self.M_A,self.M_S,self.M_R,self.bound,self.fd,self.In_T,self.In_L,self.In_A,self.In_S,self.In_R])[1,0]
                    self.In_A[0,k-list_len+1] = np.array(np.argmax(pred[k-list_len, :, :],axis=-1))
                
                elif list_len*2-1+1<= k < list_len*3-1: # Loc

                    if self.In_L[0,k-list_len*2+1] != np.zeros(2) and self.In_L[0,k-list_len*2+1] != np.zeros(code): continue
                    else: 
                        pred = MaskPLAN.predict([self.M_T,self.M_L,self.M_A,self.M_S,self.M_R,self.bound,self.fd,self.In_T,self.In_L,self.In_A,self.In_S,self.In_R])[2,0]
                        self.In_L[0,k-list_len*2+1] = np.array(np.argmax(pred[k-list_len*2, :, :],axis=-1))
                
                elif list_len*3-1+1<= k < list_len*4-1: # Size

                    if self.In_S[0,k-list_len*3+1] != 0: continue
                    else: 
                        pred = MaskPLAN.predict([self.M_T,self.M_L,self.M_A,self.M_S,self.M_R,self.bound,self.fd,self.In_T,self.In_L,self.In_A,self.In_S,self.In_R])[3,0]
                        self.In_S[0,k-list_len*3+1] = np.array(np.argmax(pred[k-list_len*3, :]))
                
                elif list_len*4-1+1<= k: # Region

                    if self.In_R[0,k-list_len*4+1] != np.zeros(4) and self.In_R[0,k-list_len*4+1] != np.zeros(code): continue
                    else: 
                        pred = MaskPLAN.predict([self.M_T,self.M_L,self.M_A,self.M_S,self.M_R,self.bound,self.fd,self.In_T,self.In_L,self.In_A,self.In_S,self.In_R])[4,0]
                        self.In_R[0,k-list_len*4+1] = np.array(np.argmax(pred[k-list_len*4, :, :],axis=-1))
                        
            # Generate imgs
                        
            if args.format == 'vec':

                # predict raw

                corners=self.In_R[1,num_room+1]
                x_min = (corners[:,0]/loc_dimen)*(bound_domain[2] - bound_domain[0]) + bound_domain[0]
                y_min = (corners[:,1]/loc_dimen)*(bound_domain[3] - bound_domain[1]) + bound_domain[1]
                x_max = (corners[:,2]/loc_dimen)*(bound_domain[2] - bound_domain[0]) + bound_domain[0]
                y_max = (corners[:,3]/loc_dimen)*(bound_domain[3] - bound_domain[1]) + bound_domain[1]
                
                for i in range(num_room):
                    pt = np.array(LineString([x_min,y_min]).envelope.exterior.coords[:-1])
                    pts.append(pt)
                    cv2.fillPoly(reconstructed, [pt[:,np.newaxis,:].astype(np.int32)], color=T_list[int(self.In_T[0,i+1])])
                    types.append(self.In_T[0,i+1])
            
                # post-process

                for m in room_order:
                    
                        if m in types:
                            
                            indexs = np.where(np.array(types)==m)[0].tolist()

                            for n in indexs:

                                if pts[n].ndim>1 and pts[n].shape[0]>2:

                                    coords = pts[n]

                                    if unioned_rooms:
                                        unioned_lines = []

                                        if unioned_rooms.geom_type == 'MultiPolygon':
                                            for g in unioned_rooms.geoms:
                                                unioned_lines.append(g.exterior)

                                        else:
                                            unioned_lines = [unioned_rooms.exterior]

                                        for t in range(4):

                                            pt1,pt2 = coords[t],coords[(t+1)%4]
                                            for unioned_line in unioned_lines:
                                                pt1,pt2 = align_target(pt1,pt2,unioned_line,3)
                                            pt1,pt2 = align_target(pt1,pt2,boundary_line,3+5)

                                            coords[t],coords[(t+1)%4] = pt1,pt2
                                        room = Polygon(coords).envelope.difference(unioned_rooms)
                                        room = room.difference(boundary_outside)
                                        if room.geom_type == 'MultiPolygon':
                                            room = room.geoms[find_max(room.geoms)]
                                        if room.geom_type == 'GeometryCollection':
                                            room = room.geoms[0]
                                        if room.area > 0.1:

                                            if len(unioned_lines) > 1:
                                                geolis = []
                                                for l in unioned_rooms.geoms:
                                                    geolis.append(l)
                                                geolis.append(room)
                                                unioned_rooms = unary_union(MultiPolygon(geolis))
                                            else:
                                                if unioned_rooms.geom_type == 'MultiPolygon':
                                                    unioned_rooms = unioned_rooms.geoms[0]
                                                unioned_rooms = unary_union(MultiPolygon([room,unioned_rooms]))
                                            
                                            cv2.fillPoly(reconst_post, [np.array(room.exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], T_list[int(types[n])])

                                    else:

                                        for t in range(4):

                                            pt1,pt2 = coords[t],coords[(t+1)%4]
                                            pt1,pt2 = align_target(pt1,pt2,boundary_line,3)
                                            coords[t],coords[(t+1)%4] = pt1,pt2

                                        unioned_rooms = Polygon(coords).difference(boundary_outside)
                                        if unioned_rooms.geom_type == 'MultiPolygon':
                                            unioned_rooms = unioned_rooms.geoms[find_max(unioned_rooms.geoms)]
                                        coorners = np.array(unioned_rooms.exterior.coords[:-1])
                                        if coorners.ndim>1 and coorners.shape[0]>2:
                                            cv2.fillPoly(reconst_post, [coorners[:,np.newaxis,:].astype(np.int32)], T_list[int(types[n])])
                        
            else: # use vq decoder to reconstruct

                priors = self.In_R[0,1:-1].copy().reshape((list_len-2,code))
                priors_ohe = tf.one_hot(priors.astype("int32"), code_dimen).numpy()
                quantized = tf.matmul(priors_ohe.astype("float32"), quantizer.embeddings, transpose_b=True)
                quantized = tf.reshape(quantized, (-1, *((5,5,code_dimen))))

                generated_samples = VQ_decoder.predict(quantized).reshape((list_len-2,128,128,4))

                for i in range(num_room):

                    # predict raw

                    predicted = generated_samples[i,:,:,:]*255
                    cond2 = ((predicted[:,:,0]>240)&(predicted[:,:,1]>240)&(predicted[:,:,2]>240)&(predicted[:,:,3]>240))
                    cull = np.where(predicted[:,:,1]<240)
                    valid = np.where(predicted[:,:,1]>239)
                    predicted[cull] = 0
                    predicted[valid] = 220

                    cv2.imwrite("Inference/output_raw_%d.png" % (i),predicted)
                    img2 = cv2.imread("Inference/output_raw_%d.png" % (i), cv2.IMREAD_UNCHANGED)

                    imgray2 = cond2*np.ones((128,128),dtype=np.uint8)*255
                    _, thresh2 = cv2.threshold(imgray2, 127, 255, 0)
                    contours2, _ = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    img2,pt2 = draw_approx_hull_polygon(img2, contours2, T_list[int(self.In_T[0,i+1])])
                    
                    reconstructed += img2

                    if pt2 != []:
                        pts.append(np.squeeze(pt2[0]))
                        types.append(self.In_T[0,i+1])

                # post-process

                for m in room_order:
                
                    if m in types:
                        
                        indexs = np.where(np.array(types)==m)[0].tolist()

                        for n in indexs:

                            if pts[n].ndim>1 and pts[n].shape[0]>2:

                                roompt = MultiPoint(pts[n])
                                rectan = roompt.envelope # make sure to be rectangular shape
                                coords = np.array(rectan.exterior.coords[:-1])

                                if unioned_rooms:
                                    unioned_lines = []

                                    if unioned_rooms.geom_type == 'MultiPolygon':
                                        for g in unioned_rooms.geoms:
                                            unioned_lines.append(g.exterior)

                                    else:
                                        unioned_lines = [unioned_rooms.exterior]

                                    for t in range(4):

                                        pt1,pt2 = coords[t],coords[(t+1)%4]
                                        for unioned_line in unioned_lines:
                                            pt1,pt2 = align_target(pt1,pt2,unioned_line,3)
                                        pt1,pt2 = align_target(pt1,pt2,boundary_line,3+5)

                                        coords[t],coords[(t+1)%4] = pt1,pt2
                                    room = Polygon(coords).envelope.difference(unioned_rooms)
                                    room = room.difference(boundary_outside)
                                    if room.geom_type == 'MultiPolygon':
                                        room = room.geoms[find_max(room.geoms)]
                                    if room.geom_type == 'GeometryCollection':
                                        room = room.geoms[0]
                                    if room.area > 0.1:

                                        if len(unioned_lines) > 1:
                                            geolis = []
                                            for l in unioned_rooms.geoms:
                                                geolis.append(l)
                                            geolis.append(room)
                                            unioned_rooms = unary_union(MultiPolygon(geolis))
                                        else:
                                            if unioned_rooms.geom_type == 'MultiPolygon':
                                                unioned_rooms = unioned_rooms.geoms[0]
                                            unioned_rooms = unary_union(MultiPolygon([room,unioned_rooms]))
                                        
                                        cv2.fillPoly(reconst_post, [np.array(room.exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], T_list[int(types[n])])

                                else:

                                    for t in range(4):

                                        pt1,pt2 = coords[t],coords[(t+1)%4]
                                        pt1,pt2 = align_target(pt1,pt2,boundary_line,3)
                                        coords[t],coords[(t+1)%4] = pt1,pt2

                                    unioned_rooms = Polygon(coords).difference(boundary_outside)
                                    if unioned_rooms.geom_type == 'MultiPolygon':
                                        unioned_rooms = unioned_rooms.geoms[find_max(unioned_rooms.geoms)]
                                    coorners = np.array(unioned_rooms.exterior.coords[:-1])
                                    if coorners.ndim>1 and coorners.shape[0]>2:
                                        cv2.fillPoly(reconst_post, [coorners[:,np.newaxis,:].astype(np.int32)], T_list[int(types[n])])

            cv2.imwrite('Inference/%s_%s_%s/iteration/raw/%d.png' % (args.model,args.deep,args.format,site_id),reconstructed)
            cv2.imwrite('Inference/%s_%s_%s/iteration/post/%d.png' % (args.model,args.deep,args.format,site_id),reconst_post)
    
    return MaskPLAN_Inference()

if __name__ == "__main__":

    # init
    args = parse_args()
    
    # run
    model = main(args)

    for site in Testset_ids[:args.test_cases]:
        model.reset()
        model.partial_input(site)
        model.inference_interation(site)
    