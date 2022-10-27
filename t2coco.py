#author: krlab-ma
#        2022.05.20
#Code Function:
#              We parse the xml infos as json file
#              type that could be training for mmdetection or DeRT.
#              Making the customize dataset as type as COCO to input to mmdetection or DeRT.
#              Please read https://cocodataset.org/#format-data to understand COCO data fomat
#   !!! please pay attention about the function cmpRbbx which extract Rotation BBox and transfer to non-Rotation BBox !!!
#                         !!! you can easily modify this code for your task !!!

#!/usr/bin/python/
#-*-coding:UTF-8-*-

import os
import cv2
import json
import math
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET

xmls = '.../xml_files'                                                         # path to xml files
imgs_pth = '.../dataset_ids'                                                   # path to train.txt val.txt test.txt, each txt file including image name
cls_id = {'blue':'0', 'orange':'1', 'red':'2'}                                 # please change the category and id following your task
save = '.../outputs'                                                           # path for outputing train.json, val.json, test.json

class xml2json:
    def __init__(self, root_pth, txt_pth, xmls_pth, cls_id, save_pth):
        self.root_pth = root_pth
        self.txt_pth = txt_pth
        self.xmls_pth = xmls_pth
        self.cls_id = cls_id
        self.save_pth = save_pth

    def converting(self):
        json_dict = {}    
        images = []
        annotations = []
        categories = []
        
        info = {
                'description':'Detection',
                'contributor':'Kochi University of Technology',
                'Date':'2022/10/25',
                'Team':'KRLAB'
        }

        for k,v in (self.cls_id).items():
            cate = {'supercategory':'None',
                    'id':v,
                    'name':k
                   }
            categories.append(cate)

        start_id = 0
        for file in os.listdir(self.txt_pth):
            txt_file = os.path.join(self.txt_pth, file)
            images, img_id = self.op2ip(images, txt_file)
            annotations = self.for_ants(annotations, txt_file, img_id)
            ants, next_id = self.add_obj_id(annotations, start_id)
            start_id = next_id

            json_dict = {'info':info,
                         'images':images,
                         'annotations':annotations,
                         'categories':ants
                        }
            stage = file.split('.')[0]
            json_pth = os.path.join(self.save_pth, 'instances_'+stage+'2017'+'.json')
            with open(json_pth, 'w', encoding='utf-8') as jsonfile:
                json.dump(json_dict, jsonfile)

    def add_obj_id(self, annotations, start_id):
        nums = len(annotations)
        new_ant = []
        for i in range(0, nums):
            ant = annotations[i]
            ant['id'] = i+start_id
            new_ant.append(ant)

        start_id = start_id+nums
        return new_ant, start_id
            
    def op2ip(self, images, txt_file):
        img_id = {}
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for idx in range(0, len(lines)):
            i = lines[idx]
            img_name = i.strip()+'.jpg'
            img_info = {'file_name':img_name,
                        'height':'512',
                        'width':'512',
                        'id':idx
                       }
            img_id[i.strip()] = idx
            images.append(img_info)

        return images, img_id

    def for_ants(self, annotations, txt_file, img_id):
        with open(txt_file) as f:
            lines = f.readlines()
        for i in lines:
            idx = img_id[i.strip()]
            xml = i.strip()+'.xml'
            xml_p = os.path.join(self.xmls_pth, xml)
            bboxes = self.parse_xml(xml_p, idx)
            for b in bboxes:
                annotations.append(b)
            
        return annotations

    def parse_xml(self, xml_file, idx):
        tree = ET.parse(xml_file)
        tree_root = tree.getroot()
        
        bboxes = []                              # there are more than one object in one image !! please pay attention
        for i in tree_root.findall('object'):
            bbox_type = i.find('type').text
            cls = i.find('name').text
            id4cls = self.cls_id[cls]
            bbox_infos = i.find(bbox_type)
            
            if bbox_type == 'robndbox':
                bbox = self.cmp_Rbbx(bbox_infos)
            elif bbox_type == 'bndbox':
                bbox = self.cmp_bbx(bbox_infos)

            area = self.cal_area(bbox)
            img_annotation = {
                              'area':area,
                              'iscrowd':'0',
                              'bbox':bbox,
                              'image_id':str(idx),
                              'category_id':id4cls
                             }
            bboxes.append(img_annotation)
       
        return bboxes

    def cal_area(self, bbox):
        w = bbox[2]
        h = bbox[3]
        area = float( float(w)*float(h) )

        return area
    
    # extracting bbox infomation from xml
    def cmp_bbx(self, bbox_infos):
        xmin = bbox_infos.find('xmin').text
        ymin = bbox_infos.find('ymin').text
        xmax = bbox_infos.find('xmax').text
        ymax = bbox_infos.find('ymax').text

        width = float(xmax)-float(xmin)
        height = float(ymax)-float(ymin)

        cx = float(xmin)+float(width)/2
        cy = float(ymin)+float(height)/2

        return [cx, cy, width, height]
 
    # extracting Rotation bbox infomation from xml
    def cmp_Rbbx(self, bbox_infos):
        cx = bbox_infos.find('cx').text
        cy = bbox_infos.find('cy').text
        w = bbox_infos.find('w').text
        h = bbox_infos.find('h').text
        radian = bbox_infos.find('angle').text
        angle = round(float(radian)*180)/math.pi
        tup = ( (float(cx), float(cy)), (float(w), float(h)), float(angle) )                 # first save to tuple
        
        rect = cv2.boxPoints(tup)                       # the compute by boxPoints, return top_left->low_left->low_right->top_right
        rect = np.array(rect)
####### follows boxPoints return value , correct way       
#        x1, y1 = rect[0]                                # top_left
#        x2, y2 = rect[3]                                # top_right
#        x3, y3 = rect[2]                                # lower_right
#        x4, y4 = rect[1]                                # lower_left
####### follows boxPoints return value , correct way 

######## error way   ##
#        min_x = np.argmin(rect[:,0])                    # min x means topleft corner
#        max_x = np.argmax(rect[:,0])                    # max x means lowerright corner
#        min_y = np.argmin(rect[:,1])                    # min y means lowerleft corner
#        max_y = np.argmax(rect[:,1])                    # max y means topright corner
        
#        x1, y1 = rect[min_x]                                # top_left
#        x2, y2 = rect[max_y]                                # top_right
#        x3, y3 = rect[max_x]                                # lower_right
#        x4, y4 = rect[min_y]                                # lower_left
####### error way    ##      
        
        min_x = np.min(rect[:,0])
        max_x = np.max(rect[:,0])
        min_y = np.min(rect[:,1])
        max_y = np.max(rect[:,1])

        regular_width = max_x - min_x
        regular_height = max_y - min_y
        
        return [str(cx), str(cy), str(regular_width), str(regular_height)]

def main():
    root = Path.cwd()
    txt_pth = os.path.join(root, imgs_pth)
    xmls_pth = os.path.join(root, xmls)
    save_pth = os.path.join(root, save)

    work = xml2json(root, txt_pth, xmls_pth, cls_id, save_pth)
    work.converting()

if __name__ == '__main__':
    main()
