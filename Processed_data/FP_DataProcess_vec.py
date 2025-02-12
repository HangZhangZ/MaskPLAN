import numpy as np
from FP_MaskPLAN_vec import *

type_dim = 10
loc_dim = 20
room_dim = 20
area_dim = 32
code = 20
duplicate = 3
num_layout = 80788

def main():

    NodeList = np.load('Processed_data/Nodes.npy',allow_pickle=True)
    NodeList[np.where(NodeList<0)] = 0
    room_corner = np.load('Processed_data/Processed_data/Room_Sparse.npy')
    pair = np.load('Processed_data/pair.npy',allow_pickle=True)

    DataSet = FloorPlan_RPlan_Dataset(type_dim, loc_dim, area_dim, room_dim, code)
    DataSet.parse_data(NodeList, pair, room_corner)
    DataSet.prepare_MaskPLAN_Data(num_layout, duplicate)

if __name__ == "__main__":

    main()