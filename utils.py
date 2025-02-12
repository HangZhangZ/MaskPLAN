import numpy as np
from tensorflow import keras
import tensorflow as tf
import cv2

def visualize_partial_input(T,L,A,S,R,path):

    canvas = cv2.imread('Partial_Input_Canvas.jpg')[:,:,0]

    for t in T: canvas[t*10:t*10+10,0:10] = 0
    for l in L: canvas[l*10:l*10+10,10:20] = 0
    for s in S: canvas[s*10:s*10+10,20:30] = 0
    for r in R: canvas[r*10:r*10+10,30:40] = 0

    for a1 in range(8):
        for a2 in range(a1,8):
            if A[a1+1,a2] == 1: 
                canvas[a2*10:a2*10+10,a1*10+50:a1*10+60] = 0
                canvas[a1*10:a1*10+10,a2*10+50:a2*10+60] = 0

    cv2.imwrite(path,canvas)

def check_room_shape(pts):
    if (pts[2,0]-pts[0,0])*(pts[2,1]-pts[0,1]) < 25:
        pts[:2,1] -= 5
        pts[2:,1] += 5
        pts[0,0] -= 5
        pts[3,0] -= 5
        pts[1:3,0] += 5
    if (pts[2,0]-pts[0,0])/(pts[2,1]-pts[0,1]) > 3:
        pts[:2,1] -= 10
        pts[2:,1] += 10
    elif (pts[2,1]-pts[0,1])/(pts[2,0]-pts[0,0]) > 3:
        pts[0,0] -= 10
        pts[3,0] -= 10
        pts[1:3,0] += 10

def randint_diff(valid): 

    if valid == 1: out = np.array([])

    else:
        n = np.random.randint(0,round(valid*0.5))
        out = np.random.choice(valid-1, n, replace=False) + 1

    return out

def randint_ada(valid): 

    if valid == 1: out = np.array([])

    else:
        n = np.random.randint(round(valid*0.5),valid)
        out = np.random.choice(valid-1, n, replace=False) + 1

    return out

def randint_room(valid): 

    if valid == 1: out = np.array([])

    else:
        n = np.random.randint(0,round(valid*0.5))
        out = np.random.choice(valid-1, n, replace=False) + 1

    return out

def mask_ada(idx,list_len,original):
    mask_init = np.zeros((list_len,list_len-2))
    mask_init[0] = 1
    if idx.shape[0] > 1:
        for m,n in enumerate(idx):
            for k,j in enumerate(idx[m+1:]):
                mask_init[n+1,j] = 1
                mask_init[j+1,n] = 1
            if m == idx.shape[0]-2:
                break
    mask_result = mask_init*original
    return mask_result

def random_mask_All_cross(T, L, A, S, R, num_layout, list_len, code): 
        
        T_mask = np.zeros((num_layout,list_len))
        L_mask = np.zeros((num_layout,list_len,code))
        A_mask = np.zeros((num_layout,list_len,list_len-2))
        S_mask = np.zeros((num_layout,list_len))
        R_mask = np.zeros((num_layout,list_len,code))

        for i in range(num_layout):

            valid = (T[i]==8).argmax(axis=0)-1
            select_T = randint_diff(valid)
            select_L = randint_diff(valid)
            select_A = randint_ada(valid)
            select_S = randint_diff(valid)
            select_R = randint_room(valid)

            T_mask[i][select_T] = T[i][select_T]
            T_mask[i][0] = T[i][0]
            L_mask[i][select_L] = L[i][select_L]
            L_mask[i][0] = L[i][0]
            S_mask[i][select_S] = S[i][select_S]
            S_mask[i][0] = S[i][0]
            R_mask[i][select_R] = R[i][select_R]
            R_mask[i][0] = R[i][0]

            A_mask[i] = mask_ada(select_A,list_len,A[i])

        return T_mask.astype(np.int32),L_mask.astype(np.int32),A_mask.astype(np.int32),S_mask.astype(np.int32),R_mask.astype(np.int32)

def random_mask_All_vec(T, L, A, S, R, num_layout, list_len, code): 
        
        T_mask = np.zeros((num_layout,list_len))
        L_mask = np.zeros((num_layout,list_len,2))
        A_mask = np.zeros((num_layout,list_len,list_len-2))
        S_mask = np.zeros((num_layout,list_len))
        R_mask = np.zeros((num_layout,list_len,code))

        for i in range(num_layout):

            valid = (T[i]==8).argmax(axis=0)-1
            select_T = randint_diff(valid)
            select_L = randint_diff(valid)
            select_A = randint_ada(valid)
            select_S = randint_diff(valid)
            select_R = randint_room(valid)

            T_mask[i][select_T] = T[i][select_T]
            T_mask[i][0] = T[i][0]
            L_mask[i][select_L] = L[i][select_L]
            L_mask[i][0] = L[i][0]
            S_mask[i][select_S] = S[i][select_S]
            S_mask[i][0] = S[i][0]
            R_mask[i][select_R] = R[i][select_R]
            R_mask[i][0] = R[i][0]

            A_mask[i] = mask_ada(select_A,list_len,A[i])

        return T_mask,L_mask,A_mask,S_mask,R_mask

def ada_sparse(adacency,graph):
    for i,j in enumerate(adacency):
        for m,n in enumerate(j):
            if (n[0] != 0 or n[1] != 0):
                graph[i][int(n[0])][int(n[1])] = 1
                graph[i][int(n[1])][int(n[0])] = 1
    
    return graph

def make_onehot(size,dimension):
    step = 1
    step_ind = 1/dimension
    size[np.where((0.001<=size) & (size<(step*1.5)))] = step_ind
    size[np.where(size<0.001)] = 0
    for i in range(dimension-3):
        size[np.where(((step*i+(step*1.5))<=size) & (size<(step*i+(step*2.5))))] = i*step_ind + 2*step_ind
    size[np.where((dimension-(step*1.5))<=size)] = 1 - step_ind
    sizes_onehot = np.array(size.copy()*dimension,dtype=np.int32)

    return sizes_onehot

def make_onehot_nor(size,dimension):
    step_ind = 1/dimension
    size[np.where((0.001<=size) & (size<(step_ind*1.5)))] = step_ind
    size[np.where(size<0.001)] = 0
    for i in range(dimension-3):
        size[np.where(((step_ind*i+(step_ind*1.5))<=size) & (size<(step_ind*i+(step_ind*2.5))))] = i*step_ind + 2*step_ind
    size[np.where((dimension-(step_ind*1.5))<=size)] = 1 - step_ind
    sizes_onehot = np.array(size.copy()*dimension,dtype=np.int32)

    return sizes_onehot

def randint_size_n(n, N=8): 
    return np.random.choice(N, n, replace=False)

