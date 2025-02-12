import numpy as np
import cv2
from shapely.geometry import LineString

max_decoded_sentence_length = 10

def decode_type(encoded,In_T,Trained_De_type):
    decoded_sentence = In_T.copy()
    for i in range(max_decoded_sentence_length):
        if decoded_sentence[:,i+1] != 0: continue
        else:
            tokenized_target_sentence = decoded_sentence[:,:-1]
            predictions = Trained_De_type.predict([tokenized_target_sentence, encoded])
            sampled_token_index = np.array(np.argmax(predictions[0, i, :]))
            decoded_sentence[:,i+1] = sampled_token_index
            if sampled_token_index == 9:
                break
    return decoded_sentence,i

def decode_loc(encoded,num_room,types_in,In_L,Trained_De_Loc):
    decoded_sentence = In_L.copy()
    for i in range(num_room+1):
        if decoded_sentence[:,i+1,0] != 0 and decoded_sentence[:,i+1,1] != 0: continue
        else:
            tokenized_target_sentence = decoded_sentence[:,:-1]
            predictions = Trained_De_Loc.predict([tokenized_target_sentence, types_in, encoded])
            sampled_token_index = np.array(np.argmax(predictions[0, i, :, :],axis=-1))
            decoded_sentence[:,i+1] = sampled_token_index

    return decoded_sentence

def decode_ada(encoded,num_room,types_in,locs_in,In_A,Trained_De_Ada):
    decoded_sentence = In_A.copy()
    for i in range(num_room+1):
        tokenized_target_sentence = decoded_sentence[:,:-1]
        predictions = Trained_De_Ada.predict([tokenized_target_sentence,types_in, locs_in, encoded])
        predictions[np.where(predictions>0.5)] = 1
        predictions[np.where(predictions<=0.5)] = 0
        sampled_token_index = np.array(predictions[0, i, :])
        decoded_sentence[:,i+1] = sampled_token_index

    return decoded_sentence

def decode_ada_new(encoded,num_room,types_in,locs_in,In_A,Trained_De_Ada):
    decoded_sentence = In_A.copy()
    for i in range(num_room+1):
        tokenized_target_sentence = decoded_sentence[:,:-1]
        predictions = Trained_De_Ada.predict([tokenized_target_sentence,types_in, locs_in, encoded])
        sampled_token_index = np.array(np.argmax(predictions[0, i, :, :],axis=-1))
        decoded_sentence[:,i+1] = sampled_token_index

    return decoded_sentence

def decode_area(encoded,num_room,types_in,locs_in,ada_in,In_S,Trained_De_Area):
    decoded_sentence = In_S.copy()
    for i in range(num_room+1):
        if decoded_sentence[:,i+1] != 0: continue
        else:
            tokenized_target_sentence = decoded_sentence[:,:-1]
            predictions = Trained_De_Area.predict([tokenized_target_sentence, types_in, locs_in, ada_in, encoded])
            sampled_token_index = np.array(np.argmax(predictions[0, i, :]))
            decoded_sentence[:,i+1] = sampled_token_index

    return decoded_sentence

def decode_room(encoded,num_room,types_in,locs_in,ada_in,area_in,In_R,Trained_De_Room):
    decoded_sentence = In_R.copy()
    for i in range(num_room+1):
        tokenized_target_sentence = decoded_sentence[:,:-1]
        predictions = Trained_De_Room.predict([tokenized_target_sentence, types_in, locs_in, ada_in, area_in, encoded])
        sampled_token_index = np.array(np.argmax(predictions[0, i, :, :],axis=-1))
        decoded_sentence[:,i+1] = sampled_token_index

    return decoded_sentence

def draw_approx_hull_polygon(img, cnts, T):
    img = np.zeros(img.shape, dtype=np.uint8)

    epsilion = img.shape[0]/128
    approxes = [cv2.approxPolyDP(cnt, epsilion, True) for cnt in cnts]
    cv2.fillPoly(img, approxes, T)

    return img,approxes

def get_bound_pt(img):

    _,thresh2 = cv2.threshold(img, 120, 255, 0)
    contours2,_ = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    epsilion = img.shape[0]/128
    approxes = [cv2.approxPolyDP(cnt, epsilion, True) for cnt in contours2]
    
    return np.squeeze(approxes[0])

def align_target(pt1,pt2,target,gap):

    vector1to2 = pt2 - pt1
    norl_vec1to2 = vector1to2/np.linalg.norm(vector1to2)
    vector2to1 = pt1 - pt2
    norl_vec2to1 = vector2to1/np.linalg.norm(vector2to1)

    extended_pt1 = pt1 + norl_vec2to1*gap
    extended_pt2 = pt2 + norl_vec1to2*gap

    extended_pt1 = LineString([extended_pt1.tolist(),pt1.tolist()])
    extended_pt2 = LineString([extended_pt2.tolist(),pt2.tolist()])

    try:
        inter_pt1 = extended_pt1.intersection(target)
        inter_pt2 = extended_pt2.intersection(target)

        if inter_pt1.geom_type == 'Point':
            pt1 = np.array(inter_pt1.coords[:][0]).astype(np.int32)
        if inter_pt2.geom_type == 'Point':
            pt2 = np.array(inter_pt2.coords[:][0]).astype(np.int32)
    except:
        pass

    return pt1,pt2

def find_max(multi_geo):
    areas = []
    for m in multi_geo:
        areas.append(m.area)
    idx = areas.index(max(areas))

    return idx
