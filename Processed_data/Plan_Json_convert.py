import numpy as np
import json

# All Values in Meter scale
# Type_List: max 8 rooms in one layout --> max list len == 8
# 0 - None, 1 - Living, 2 - Bath, 3 - Closet, 4 - Bed, 5 - Kitchen, 6 - Dining, 7 - Balcony
# Loc_x_List/Loc_y_List: central point --> [xx.xx m,xx.xx m]
# Area_List: area real size --> xx.xx m2
# Ratio_List: room shape --> room height / room width
# Graph_List: layout graph --> 8*8 sparse matrix, 1 == connected, 0 == no-adjacent
# Bound: vectorized Bound edges --> [[xx.xx m,xx.xx m],[xx.xx m,xx.xx m]...]

name_list = ['None','Living','Bath','Closet','Bed','Kitchen','Dining','Balcony']

def parse_rooms(Type_List,Loc_x_List,Loc_y_List,Area_List,Ratio_List):
    room_area_list = []
    room_ratio_list = []
    room_num = 0
    rest_room = 0
    Kitchen_S = 0
    Kitchen_R = 0
    Kitchen_x = 0
    Kitchen_y = 0
    Kitchen_S = 0
    Kitchen_R = 0
    Kitchen_x = 0
    Kitchen_y = 0
    for j,k in enumerate(Type_List):
        if k != 0 and k!= 1:
            rest_room += 1
        if k == 4 or k == 5 or k == 6:
            room_area_list.append(Area_List[j])
            room_ratio_list.append(Ratio_List[j])
            room_num += 1
        if k == 5:
            Kitchen_S = room_area_list[j]
            Kitchen_R = room_ratio_list[j]
            Kitchen_x = Loc_x_List[j]
            Kitchen_y = Loc_y_List[j]
        if k == 6:
            Dining_S = room_area_list[j]
            Dining_R = room_ratio_list[j]
            Dining_x = Loc_x_List[j]
            Dining_y = Loc_y_List[j]
    
    return room_num,rest_room,Dining_x,Dining_y,Dining_S,Dining_R,Kitchen_x,Kitchen_y,Kitchen_S,Kitchen_R

def Write_Json(filename,Type_List,Loc_x_List,Loc_y_List,Area_List,Ratio_List,Graph_List,Bound,Door):
    
    room_num,all_room,Dining_x,Dining_y,Dining_S,Dining_R,Kitchen_x,Kitchen_y,Kitchen_S,Kitchen_R = parse_rooms(Type_List,Loc_x_List,Loc_y_List,Area_List,Ratio_List)

    data = {
            "Room_Types": Type_List,
            "Loc_x_List": Loc_x_List,
            "Loc_y_List": Loc_y_List,
            # "Loc_z_List": Loc_z_List, indicating floors
            "Room_Sizes": Area_List,
            "Room_Shapes": Ratio_List,
            "adjacent_Graph": Graph_List,
            "Bound_Corners": Bound,
            "Front_Door": Door,
            # for rooms evaluation
            "Room_Count": room_num,
            "Room_All": all_room,
            # for Kitchen and Dining evaluation
            "Dining_x": Dining_x,
            "Dining_y": Dining_y,
            "Dining_Size": Dining_S,
            "Dining_Shape": Dining_R,
            "Kitchen_x": Kitchen_x,
            "Kitchen_y": Kitchen_y,
            "Kitchen_Size": Kitchen_S,
            "Kitchen_Shape": Kitchen_R,
            }
    
    with open(filename, "w") as f:
        json.dump(data, f)
    
def Rplan_to_Json(filename,Type_List,Loc_x_List,Loc_y_List,Area_List,Ratio_List,Graph_List,Bound,Door):

    data = {
        "Edges": Graph_List.tolist(),
        "boundary_Corners": Bound.tolist(),
        "Front_Door": Door.tolist(),
        "nodes": [
            {
                "name": name_list[types],
                "id":type,
                "area": Area_List[types].tolist(),
                "Shape": Ratio_List[types].tolist(),
                "position": {
                    "x": Loc_x_List[types].tolist(),
                    "y": Loc_y_List[types].tolist(),
                    "z": 0.0
                },
                "angle": 0.0,
            }
            for types,type in enumerate(Type_List) if type != 0
        ]
    }
    with open(filename, "w") as f:
        json.dump(data, f)
