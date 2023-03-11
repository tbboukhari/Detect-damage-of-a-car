from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import os
def config1 (img):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("/Users/mac/Downloads/car-detect/models/part_segmentation_model.pth"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (damage) + 1
    cfg.MODEL.RETINANET.NUM_CLASSES = 2 # only has one class (damage) + 1
    cfg.MODEL.WEIGHTS = os.path.join('/Users/mac/Desktop/car-detect/models/part_segmentation_model.pth')
    #/content/drive/MyDrive/archive/damage_segmentation_model.pth
    # /content/drive/MyDrive/archive/part_segmentation_model.pth
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
    cfg['MODEL']['DEVICE']='cpu'#or cpu
    part_predictor = DefaultPredictor(cfg)
    return(part_predictor)