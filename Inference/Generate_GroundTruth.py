import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from Inference.decode_function import get_bound_pt, draw_approx_hull_polygon
import argparse
from shapely.geometry import Polygon, LinearRing

from MaskPLAN.MaskPLAN import VectorQuantizer

## Global Para ##

list_len = 10
type_dimen = 10
code = 25
code_dimen = 64

# color tag for each room type, just for visualization
T_list = [[255,255,255,255],[255,255,0,255],[255,0,255,255],[0,255,255,255],
          [0,0,255,255],[255,0,0,255],[0,255,0,255],[127,127,255,255],[127,255,127,255]]

## Load Data ##

Testset_ids = np.load('Processed_data/Test_set.npy')

## args ##

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--format', default='cross', type=str, help='cross or vec')
    parser.add_argument('--test_cases', default=1000, type=int)

    return parser.parse_args()

## main ##

def main(args):

    os.makedirs('Inference/GroundTruth_%s/raw' % (args.format), exist_ok=True)

    # load data based on format
    if args.format == 'cross':
        Input_data = np.load('Processed_data/RPLAN_input_cross.npz')
    else:
        Input_data = np.load('Processed_data/RPLAN_input_vec.npz')

    file_index = Input_data.files
    T_in = Input_data[file_index[0]]
    R_in = Input_data[file_index[4]]

    # load VQVAE for cross format
    if args.format == 'cross':
        VQ_decoder = load_model('VQ_Pretrained/mix_5564/vqvae_de/decoder.keras')
        VQ_value = np.load('VQ_Pretrained/mix_5564/vqvae_q/quantizer.npy')
        quantizer = VectorQuantizer(code_dimen, code_dimen)
        quantizer.embeddings = tf.Variable(initial_value=VQ_value, trainable=False, name="embeddings_vqvae_R")

    for num, site_id in enumerate(Testset_ids[:args.test_cases]):

        # get number of rooms
        num_room = (T_in[site_id] == type_dimen - 2).argmax(axis=0) - 1
        if num_room <= 0:
            num_room = list_len - 2

        # get boundary
        boundary = cv2.imread('parsed_img/img_room_sqe/0/%d.png' % (site_id), cv2.IMREAD_UNCHANGED)[:,:,-1]
        boundary[np.where(boundary > 100)] = 255
        boundary_pt = get_bound_pt(boundary)
        if boundary_pt.ndim == 1:
            boundary_pt = [[0,0],[0,127],[127,127],[127,0]]
        boundary_line = LinearRing(boundary_pt)

        # fill boundary in img
        reconstructed = np.zeros((128,128,4))
        cv2.fillPoly(reconstructed, [np.array(boundary_line.buffer(2).exterior.coords[:-1])[:,np.newaxis,:].astype(np.int32)], color=[255]*4)
        cv2.fillPoly(reconstructed, [np.array(boundary_line.coords[:-1])[:,np.newaxis,:].astype(np.int32)], color=[0]*4)

        if args.format == 'cross':
            # use vq decoder to reconstruct ground truth rooms
            priors = R_in[site_id, 1:-1].copy().reshape((list_len-2, code))
            priors_ohe = tf.one_hot(priors.astype("int32"), code_dimen).numpy()
            quantized = tf.matmul(priors_ohe.astype("float32"), quantizer.embeddings, transpose_b=True)
            quantized = tf.reshape(quantized, (-1, *((5,5,code_dimen))))
            generated_samples = VQ_decoder.predict(quantized, verbose=0).reshape((list_len-2, 128, 128, 4))

            for i in range(num_room):
                predicted = generated_samples[i,:,:,:] * 255
                cond2 = ((predicted[:,:,0] > 240) & (predicted[:,:,1] > 240) & (predicted[:,:,2] > 240))
                cull = np.where(predicted[:,:,1] < 240)
                valid = np.where(predicted[:,:,1] > 239)
                predicted[cull] = 0
                predicted[valid] = 220

                cv2.imwrite("Inference/output_raw_%d.png" % (i), predicted)
                img2 = cv2.imread("Inference/output_raw_%d.png" % (i), cv2.IMREAD_UNCHANGED)

                imgray2 = cond2 * np.ones((128,128), dtype=np.uint8) * 255
                _, thresh2 = cv2.threshold(imgray2, 127, 255, 0)
                contours2, _ = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img2, pt2 = draw_approx_hull_polygon(img2, contours2, T_list[int(T_in[site_id, i+1])])

                reconstructed += img2

        else:
            # vec format: use room corner coordinates directly
            bound_domain = np.load('Processed_data/Boundary_BoundingBox.npy')
            loc_dimen = 20
            from shapely.geometry import LineString

            corners = R_in[site_id, 1:num_room+1]
            x_min = ((corners[:,0]+1)/loc_dimen)*(bound_domain[site_id,3] - bound_domain[site_id,1]) + bound_domain[site_id,1]
            y_min = ((corners[:,1]+1)/loc_dimen)*(bound_domain[site_id,2] - bound_domain[site_id,0]) + bound_domain[site_id,0]
            x_max = ((corners[:,2]+1)/loc_dimen)*(bound_domain[site_id,3] - bound_domain[site_id,1]) + bound_domain[site_id,1]
            y_max = ((corners[:,3]+1)/loc_dimen)*(bound_domain[site_id,2] - bound_domain[site_id,0]) + bound_domain[site_id,0]

            for i in range(num_room):
                pt = np.array(LineString([[y_min[i],x_min[i]],[y_max[i],x_max[i]]]).envelope.exterior.coords[:-1])
                cv2.fillPoly(reconstructed, [pt[:,np.newaxis,:].astype(np.int32)], color=T_list[int(T_in[site_id, i+1])])

        cv2.imwrite('Inference/GroundTruth_%s/raw/%d.png' % (args.format, site_id), reconstructed)

        if num % 100 == 0:
            print('Processed %d samples' % num)

    print('Ground truth generation complete!')

if __name__ == "__main__":
    args = parse_args()
    main(args)
