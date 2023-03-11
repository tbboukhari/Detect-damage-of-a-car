from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from Config_damage_part import config1
from Config_damage_damage import config2
# im = io.imread("/Users/reezocar/Desktop/car-detect/archive/52796_1.jpeg")


def part_outputs() :
#part inference
    parts_outputs = config1(im)
    parts_v = Visualizer(im[:, :, ::-1],
                    metadata=MetadataCatalog.get("car_mul_dataset_val"), 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    parts_out = parts_v.draw_instance_predictions(parts_outputs["instances"].to("cpu"))
    return parts_outputs
