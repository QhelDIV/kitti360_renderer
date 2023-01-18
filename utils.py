from __future__ import print_function, absolute_import, division
import os
import json
from skimage import io, filters
import numpy as np
from collections import namedtuple
from collections import defaultdict
from matplotlib import cm
import xml.etree.ElementTree as ET
import glob
import struct

# get current date and time
import datetime
import locale

# A point in a polygon
Point = namedtuple('Point', ['x', 'y'])


from abc import ABCMeta, abstractmethod
#from kitti360scripts.helpers.labels     import labels, id2label, kittiId2label, name2label
from labels    import labels, id2label, kittiId2label, name2label
MAX_N = 10000
def local2global(semanticId, instanceId):
    globalId = semanticId*MAX_N + instanceId
    if isinstance(globalId, np.ndarray):
        return globalId.astype(np.int)
    else:
        return int(globalId)

def global2local(globalId):
    semanticId = globalId // MAX_N
    instanceId = globalId % MAX_N
    if isinstance(globalId, np.ndarray):
        return semanticId.astype(np.int), instanceId.astype(np.int)
    else:
        return int(semanticId), int(instanceId)

annotation2global = defaultdict()

import sys
import glob
import numpy as np
from kitti360scripts.helpers.annotation import Annotation3D, KITTI360Bbox3D

class Annotation3D_fixed(Annotation3D):
    # override the init_instance function, to use the override KITTI360Bbox3D.parseBbox function
    def init_instance(self, labelPath):
        # load annotation
        tree = ET.parse(labelPath)
        root = tree.getroot()

        self.objects = defaultdict(dict)
        self.objcount = dict()

        self.num_bbox = 0

        for child in root:
            if child.find('transform') is None:
                continue
            obj = KITTI360Bbox3D_fixed() # override the KITTI360Bbox3D.parseBbox function
            obj.parseBbox(child) # override the KITTI360Bbox3D.parseBbox function
            if obj.semanticId not in self.objcount:
                self.objcount[obj.semanticId] = 0
            if obj.instanceId==-2:
                obj.instanceId = self.objcount[obj.semanticId]
            self.objcount[obj.semanticId] += 1
            
            globalId = local2global(obj.semanticId, obj.instanceId)
            self.objects[globalId][obj.timestamp] = obj
            self.num_bbox+=1

        globalIds = np.asarray(list(self.objects.keys()))
        semanticIds, instanceIds = global2local(globalIds)
        for label in labels:
            if label.hasInstances:
                print(f'{label.name:<30}:\t {(semanticIds==label.id).sum()}')
        print(f'Loaded {len(globalIds)} instances')
        print(f'Loaded {self.num_bbox} boxes')
classmap = {'driveway': 'parking', 'ground': 'terrain', 'unknownGround': 'ground', 
                    'railtrack': 'rail track', 'trashbin':"unlabeled"}
class KITTI360Bbox3D_fixed(KITTI360Bbox3D):
    def parseBbox(self, child):
        # try to catch all errors during parsing
        self.labelname = child.find('label').text
        if self.labelname in classmap:
            self.labelname = classmap[self.labelname]
        # try:
        #     semanticIdKITTI = int(child.find('semanticId').text)
        #     self.semanticId = kittiId2label[semanticIdKITTI].id
        #     self.instanceId = int(child.find('instanceId').text)
        #     #self.name = kittiId2label[semanticIdKITTI].name
        # except:
        self.semanticId = name2label[self.labelname].id
        self.instanceId = -2

        self.start_frame = int(child.find('start_frame').text)
        self.end_frame = int(child.find('end_frame').text)

        self.timestamp = int(child.find('timestamp').text)

        self.annotationId = int(child.find('index').text) + 1

        global annotation2global
        annotation2global[self.annotationId] = local2global(self.semanticId, self.instanceId)
        self.parseVertices(child)
# %%
import pandas as pd
import glob
import numpy as np
def count_all_labels(labelDir):
    labelPaths = glob.glob(os.path.join(labelDir, '*.xml'))
    dframes = []
    for labelPath in labelPaths:
        dframes.append( pd.read_xml(labelPath) )
    dframe = pd.concat(dframes)
    dlabels = dframe.label.to_numpy()
    # also return count of each label
    print(np.unique(dlabels, return_counts=True))
    uniquel = np.unique(dlabels)
    print(dframe["label"].value_counts())
    return dframe
    #annotation = Annotation3D_fixed(labelPath)
    #print(annotation.num_bbox)
dframe = count_all_labels("/localhome/xya120/studio/sherwin_project/KITTI-360/3d_bboxes_full/train_full")

# %%
