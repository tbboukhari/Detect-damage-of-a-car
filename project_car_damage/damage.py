from unittest import result
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from Config_damage_part import config1
from Config_damage_damage import config2

# im = io.imread("/Users/reezocar/Desktop/car-detect/archive/52796_1.jpeg")

def damage_outputs():
    damage = config2(im)
    damage_v = Visualizer(im[:, :, ::-1],
                    metadata=MetadataCatalog.get("car_dataset_val"), 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    damage_out = damage_v.draw_instance_predictions(damage["instances"].to("cpu"))
    return result

