

from unittest import result
from Config_damage_damage import config2
from damage import damage_outputs
# from part import part_outputs
from detect_damage_part import detect_damage_parties
from part import part_outputs
import imageio as iio

# im = iio.imread("/Users/reezocar/Desktop/car-detect/archive/52796_1.jpeg")


def car_state() :
    parts_dict = {}
    damage_outputs = config2(im)
    damage_class_map= {0:'damage', 1:'not damaged'}
    parts_class_map={0:'headlamp',1:'rear_bumper', 2:'door', 3:'hood', 4: 'front_bumper'}
    damage_prediction_classes = [ damage_class_map[el] + "_" + str(indx) for indx,el in enumerate(damage_outputs["instances"].pred_classes.tolist())]
    damage_polygon_centers = damage_outputs["instances"].pred_boxes.get_centers().tolist()
    damage_dict = dict(zip(damage_prediction_classes,damage_polygon_centers))
    print("Damaged Parts: ",detect_damage_parties(damage_dict,parts_dict))
    if len(damage_dict)==0:  
        print('the car is not damaged')
    else:
        print('the car is damaged')
    return len(damage_dict)==0
car_state_res = car_state()






# def car_state():

#     damage_class_map= {0:'damage', 1:'not damaged'}
#     parts_class_map={0:'headlamp',1:'rear_bumper', 2:'door', 3:'hood', 4: 'front_bumper'}
#     damage_prediction_classes = [ damage_class_map[el] + "_" + str(indx) for indx,el in enumerate(damage["instances"].pred_classes.tolist())]
#     damage_polygon_centers = damage_outputs["instances"].pred_boxes.get_centers().tolist()
#     damage_dict = dict(zip(damage_prediction_classes,damage_polygon_centers))
#     parts_prediction_classes = [ parts_class_map[el] + "_" + str(indx) for indx,el in enumerate(part_outputs["instances"].pred_classes.tolist())]
#     parts_polygon_centers =  part_outputs["instances"].pred_boxes.get_centers().tolist()
#     #Remove centers which lie in beyond 800 units
#     parts_polygon_centers_filtered = list(filter(lambda x: x[0] < 800 and x[1] < 800, parts_polygon_centers))
#     parts_dict = dict(zip(parts_prediction_classes,parts_polygon_centers_filtered))
#     print("Damaged Parts: ",detect_damage_parties(damage_dict,parts_dict))
#     if len(damage_dict)==0:  
#         print('the car is not damaged')
#     else:
#         print('the car is damaged')
#     return detect_damage_parties(damage_dict,parts_dict)
# car_state_res  = car_state()
    