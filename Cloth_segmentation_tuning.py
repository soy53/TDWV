#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os, distutils.core


# In[2]:


run install_detectron2.ipynb


# In[3]:


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import torch as nn

from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode


# In[4]:


nn.cuda.empty_cache()


# In[5]:


from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

register_coco_instances('fashion_train', {}, 'Fashionpedia/instances_atrributes_train2020_new.json', 'Fashionpedia/train_img/')
register_coco_instances('fashion_val', {}, 'Fashionpedia/instances_attributes_val2020_new.json', 'Fashionpedia/test_img/')

#register_coco_instances('fashion_train_revised', {}, 'Fashionpedia/new_annotations.json', 'Fashionpedia/train_img/')
#register_coco_instances('fashion_val_revised', {}, 'Fashionpedia/new_annotations_val.json', 'Fashionpedia/test_img/')


#from detectron2.data import get_detection_dataset_dicts
#from detectron2.data.datasets import builtin_meta


# In[7]:


#DatasetCatalog.register = 


# In[6]:


fashion_metadata = MetadataCatalog.get("fashion_train")
dataset_dicts = DatasetCatalog.get('fashion_train')


# In[8]:


#fashion_metadata


# In[9]:


# dataset_dicts[1]["file_name"]


# In[10]:


# fname = 'Fashionpedia/train_img/0081580df39aa7cc74f70ede71460984.jpg'
# img = cv.imread(fname)
# img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
# visualizer = Visualizer(img[:, :, ::-1], metadata=fashion_metadata, scale=1)
# #vis = visualizer.draw_dataset_dict(dataset_dicts[1])
# plt.figure(figsize=(8,8))
# plt.imshow(vis.get_image()[:, :, ::-1])


# In[7]:


import random

testlist = []

for d in random.sample(dataset_dicts, 3):
    testlist.append(d)


# In[8]:



gt_list = []

for d in testlist:

    img = cv.imread(d["file_name"])
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    visualizer = Visualizer(img[:, :, ::-1], metadata=fashion_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    plt.figure(figsize=(16,16))
    gt_list.append(vis.get_image()[:, :, ::-1])
    plt.imshow(gt_list[-1])
    


# In[13]:


#dataset_dicts[3]


# In[14]:


#plt.imshow(a)


# In[15]:


########### Lets make a class


# In[9]:


from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime


# In[10]:


import logging


# In[11]:


from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)


# In[12]:


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=30,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)
        
#https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b#file-lossevalhook-py


# In[13]:


class fashionTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks


# In[14]:


import detectron2.structures
from detectron2.structures import pairwise_iou


# In[22]:


# bboxes_gt = structures.Boxes(torch.Tensor(bboxes_gt))
# bboxes_pred = outputs["instances"].pred_boxes
# IOUs = structures.pairwise_iou(bboxes_gt, bboxes_pred)


# In[15]:


from detectron2 import model_zoo

cfg = get_cfg()


# In[16]:


#cfg.merge_from_file("detectron2/configs/Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml")
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("detectron2/configs/Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml")


# In[17]:


#cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("fashion_train",)
cfg.DATASETS.TEST = ('fashion_val',)  
#cfg.DATASETS.TEST = ('fashion2_val',)  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 1


############################모델패스############################################
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "log/22_model_0004999.pth")
############################모델패스############################################

#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0000199.pth") 
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    4
)  # faster, and good enough for this toy dataset
cfg.INPUT.MIN_SIZE_TRAIN = (500,)


###############################################################################
########################### 조정하면서 관찰하셔야 하는 수치들 ####################
###############################################################################

cfg.SOLVER.BASE_LR = 0.0001

### lms상에서 10000 iter 넘어가면서부터 ssh 에러 확률 상승
### 3000 = 9시간
cfg.SOLVER.MAX_ITER = (
    25000
)  # 300 iterations seems good enough, but you can certainly train longer
####새 annotation 적용 시 23 (정도, 에러나면 수정 직접 해주세요 ㅅㅁㅅ)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 23    #47
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 23    #47
cfg.INPUT.MASK_FORMAT='bitmask'

### 다음 두 수치는 숫자를 맞춰주세요
### eval은 7분정도 걸립니다, 너무 자주하면 안좋겠지만, 너무 띄엄띄엄해도 볼 수 있는 수치가 없겠죠?
### 500 정도면 괜찮을거 같은데, train iteration에 맞춰서 조정 부탁드리겠슴다
cfg.TEST.EVAL_PERIOD = 1000
cfg.SOLVER.CHECKPOINT_PERIOD = 1000

###############################################################################

#cfg.SOLVER.GAMMA = 0.1
#cfg.SOLVER.STEPS = (15000,)

# Freeze the first several stages so they are not trained.
# There are 5 stages in ResNet. The first is a convolution, and the following
# stages are each group of residual blocks
cfg.MODEL.BACKBONE.FREEZE_AT = 5



os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


# In[26]:


#cfg


# In[27]:


#cfg.OUTPUT_DIR


# In[28]:


#pwd


# In[ ]:





# In[18]:


trainer = fashionTrainer(cfg)
#trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


# In[32]:


# import json
# import matplotlib.pyplot as plt

# experiment_folder = './output/model_iter4000_lr0005_wf1_date2020_03_20__05_16_45'

# def load_json_arr(json_path):
#     lines = []
#     with open(json_path, 'r') as f:
#         for line in f:
#             lines.append(json.loads(line))
#     return lines

# experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')

# plt.plot(
#     [x['iteration'] for x in experiment_metrics], 
#     [x['total_loss'] for x in experiment_metrics])
# plt.plot(
#     [x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
#     [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
# plt.legend(['total_loss', 'validation_loss'], loc='upper left')
# plt.show()


# In[ ]:





# In[33]:


#with open("mycfg.yaml", "w") as f: 
#    f.write(cfg.dump())


# In[34]:



# import json
# import matplotlib.pyplot as plt

# experiment_folder = './output/model_iter4000_lr0005_wf1_date2020_03_20__05_16_45'

# def load_json_arr(json_path):
#     lines = []
#     with open(json_path, 'r') as f:
#         for line in f:
#             lines.append(json.loads(line))
#     return lines

# experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')

# plt.plot(
#     [x['iteration'] for x in experiment_metrics], 
#     [x['total_loss'] for x in experiment_metrics])
# plt.plot(
#     [x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
#     [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
# plt.legend(['total_loss', 'validation_loss'], loc='upper left')
# plt.show()


# In[39]:


from detectron2.engine import DefaultPredictor
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yamlcfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") 
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0001999.pth") 
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0000999.pth") 
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0001499.pth") 
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0019999.pth") 


# In[19]:


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "log/23_model_0024999.pth")
#가운데가 0.5
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)


# In[20]:


#import cv2 as cv
test_img = cv.imread("segmentation/yn_01.png")
outputs = predictor(test_img)


# In[21]:


v = Visualizer(test_img[:, :, ::-1], metadata=fashion_metadata, scale=1.5, instance_mode=ColorMode.IMAGE_BW)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
ee = v.get_image()
plt.figure(figsize = (24,24))
plt.imshow(ee)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# import random

# for d in testlist:

#     img = cv.imread(d["file_name"])
#     outputs = predictor(img)
    
#     v = Visualizer(img[:, :, ::-1], metadata=fashion_metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
#     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     ee = v.get_image()
#     plt.figure(figsize=(16,16))
#     plt.imshow(ee)


# In[ ]:





# In[ ]:


# outputs


# In[ ]:


# type(predictor)


# In[ ]:


# mpath = "output/model_final_6ba57e.pkl"
# mpath2 = "output/model_final.pth"


# In[ ]:


# import pickle


# In[ ]:


# mpath


# In[ ]:


# pretrained = ''


# In[ ]:


# with open("output/model_final_6ba57e.pkl", 'rb') as pickle_file:
#     pretrained = pickle.load(pickle_file)


# In[ ]:


# type(pretrained)


# In[ ]:


# pretrained


# In[ ]:


#### cfg.merge_from_file("fashionpedia_tools/config.yaml")


# In[ ]:





# In[ ]:


# v = v.draw_instance_predictions(out2["instances"].to("cpu"))


# In[ ]:


# model = nn.load(mpath2)


# In[ ]:





# In[ ]:





# In[ ]:


# import fashionpedia_tools


# In[ ]:


# from fashionpedia_tools import customized


# In[ ]:


# from fashionpedia_tools import data, solver


# In[ ]:


# from detectron2.config import LazyConfig as L

# cfg = L.load("fashionpedia_tools/config.yaml")


# In[ ]:


# cfg


# In[ ]:


# cfg.MODEL.WEIGHTS = "output/model_final_6ba57e.pkl"


# In[ ]:





# In[ ]:





# In[ ]:


# predictor = DefaultPredictor(cfg)


# In[ ]:





# In[ ]:


# from detectron2.config import instantiate

# model = instantiate(cfg.model)
# optimizer = instantiate(cfg.optimizer)


# In[ ]:





# In[ ]:


# from fashionpedia_tools import customized


# In[ ]:




# predictor = DefaultPredictor(cfg)


# In[ ]:


# import logging
# from tabnanny import check
# import torch
# import gc
# from detectron2.config import instantiate
# from detectron2.utils import comm
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.engine import (
#     AMPTrainer,
#     default_setup,
#     default_writers,
#     hooks)
# from customized import EarlyStop, inference

# def do_test(cfg, model):
#     # Test function called by the EvalHook when current_iter = eval_period
#     if "evaluator" in cfg.dataloader:
#         ret = inference(
#             model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
#         )
#         return ret


# In[ ]:





# In[ ]:


# run fashionpedia_tools/main.py


# In[ ]:




