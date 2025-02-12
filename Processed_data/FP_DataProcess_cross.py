import numpy as np
from FP_MaskPLAN_cross import *

type_dim = 10
area_dim = 32
code = 25
duplicate = 3
num_layout = 80788

def main():

    NodeList = np.load('Processed_data/Nodes.npy',allow_pickle=True)
    NodeList[np.where(NodeList<0)] = 0
    pair = np.load('Processed_data/pair.npy',allow_pickle=True)
    loc_code = np.load('Processed_data/RPLAN_L_code.npy')
    room_code = np.load('Processed_data/RPLAN_R_code.npy')

    DataSet = FloorPlan_RPlan_Dataset(type_dim, area_dim, code)
    DataSet.parse_data(NodeList, loc_code, pair, room_code)
    DataSet.prepare_MaskPLAN_Data(num_layout, duplicate)

if __name__ == "__main__":

    main()