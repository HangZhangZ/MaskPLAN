import numpy as np
from utils import *

class FloorPlan_RPlan_Dataset():
    def __init__(self,type_dimen,loc_dimen,area_dimen,room_dimen,code):

        self.type_dimen = type_dimen
        self.area_dimen = area_dimen
        self.loc_dimen = loc_dimen
        self.room_dimen = room_dimen
        self.code = code

        self.RPlan_adjacency = np.zeros((80788,8,8),dtype=np.int32)

        # 0 - None, 1 - Living, 2 - Bath, 3 - Closet, 4 - Bed, 5 - Kitchen, 6 - Dining, 7 - Balcony, 8 - End, 9 - Begin
        self.RPlan_type = None

    def parse_data(self, NodeList, adacency, room_parse):

        # Type
        self.RPlan_type = NodeList[:,:,4]

        # Location
        self.RPlan_location = make_onehot_nor(NodeList[:,:,:2],self.loc_dimen)

        # Ada
        self.RPlan_adjacency = ada_sparse(adacency,self.RPlan_adjacency)
        
        # Area
        self.RPlan_area = make_onehot(NodeList[:,:,2],self.area_dimen-2)

        # Region
        self.RPlan_region = make_onehot_nor(room_parse,self.room_dimen-2)

    def prepare_MaskPLAN_Data(self, num_layout, loops):

        self.RPlan_type = np.concatenate((np.ones((num_layout,1))*(self.type_dimen-1),self.RPlan_type,np.zeros((num_layout,1))),axis=-1)
        self.RPlan_location = np.concatenate((np.ones((num_layout,1,2))*(self.loc_dimen-1),self.RPlan_location,np.zeros((num_layout,1,2))),axis=-2)
        self.RPlan_area = np.concatenate((np.ones((num_layout,1))*(self.area_dimen-1),self.RPlan_area,np.zeros((num_layout,1))),axis=-1)
        self.RPlan_adjacency = np.concatenate((np.ones((num_layout,1,8)),self.RPlan_adjacency,np.zeros((num_layout,1,8))),axis=-2)
        self.RPlan_region = np.concatenate((np.ones((num_layout,1,4))*(self.room_dimen-1),self.RPlan_region,np.zeros((num_layout,1,4))),axis=-2)
        
        for m in range(num_layout):
            end_id = np.where(self.RPlan_type[m]==0)[0][0]
            self.RPlan_type[m,end_id] = self.type_dimen-2
            self.RPlan_location[m,end_id] = np.array([self.loc_dimen-2]*2)
            self.RPlan_adjacency[m,end_id] = np.array([0,1]*4)
            self.RPlan_area[m,end_id] = np.array([self.area_dimen-2])
            self.RPlan_region[m,end_id] = np.array([self.room_dimen-2]*4)

        T_mask = np.zeros((num_layout*loops,10))
        L_mask = np.zeros((num_layout*loops,10,2))
        A_mask = np.zeros((num_layout*loops,10,8))
        S_mask = np.zeros((num_layout*loops,10))
        R_mask = np.zeros((num_layout*loops,10,4))

        T_new = np.concatenate([self.RPlan_type]*loops)
        L_new = np.concatenate([self.RPlan_location]*loops)
        A_new = np.concatenate([self.RPlan_adjacency]*loops)
        S_new = np.concatenate([self.RPlan_area]*loops)
        R_new = np.concatenate([self.RPlan_region]*loops)

        for n in range(loops):
            T_mask[n*num_layout:(n+1)*num_layout],L_mask[n*num_layout:(n+1)*num_layout],A_mask[n*num_layout:(n+1)*num_layout],S_mask[n*num_layout:(n+1)*num_layout],R_mask[n*num_layout:(n+1)*num_layout]\
            = random_mask_All_vec(self.RPlan_type, self.RPlan_location, self.RPlan_adjacency, self.RPlan_area, self.RPlan_region, num_layout, 10, 4)

        T_ou = np.concatenate((T_new[:,1:],np.zeros((num_layout*loops,1))),axis=-1).astype(np.int32)
        L_ou = np.concatenate((L_new[:,1:],np.zeros((num_layout*loops,1,2))),axis=-2).astype(np.int32)
        A_ou = np.concatenate((A_new[:,1:],np.zeros((num_layout*loops,1,8))),axis=-2).astype(np.int32)
        S_ou = np.concatenate((S_new[:,1:],np.zeros((num_layout*loops,1))),axis=-1).astype(np.int32)
        R_ou = np.concatenate((R_new[:,1:],np.ones((num_layout*loops,1,4))),axis=-2).astype(np.int32)

        np.savez_compressed('Processed_data/RPLAN_input_vec.npz',T=T_new,L=L_new,A=A_new,S=S_new,R=R_new)
        np.savez_compressed('Processed_data/RPLAN_output_vec.npz',T=T_ou,L=L_ou,A=A_ou,S=S_ou,R=R_ou)
        np.savez_compressed('Processed_data/RPLAN_mask_vec.npz',T=T_mask,L=L_mask,A=A_mask,S=S_mask,R=R_mask)

        

        


