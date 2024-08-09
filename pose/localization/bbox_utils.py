from pose.localization.bbox import BoundingBox
import numpy as np
from shapely.geometry import Polygon


def distance(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
   return np.sqrt(((bbox1.center[0] - bbox2.center[0]) ** 2) + 
                  ((bbox1.center[1] - bbox2.center[1]) ** 2))


def get_food_bboxes(bounding_boxes: list[BoundingBox]):
    bboxes = [bbox_obj for bbox_obj in bounding_boxes if bbox_obj.label == 'food']
    return bboxes


def get_closest_food_bbox(mouth_bbox: BoundingBox, food_bboxes: list[BoundingBox]) -> BoundingBox:
    bbox_dists = []
    for bbox in food_bboxes:
        dist = distance(mouth_bbox, bbox)
        bbox_dists.append([dist, bbox])

    bbox_dists.sort()
    return bbox_dists[0][1]


def get_mouth_bbox(bounding_boxes: list[BoundingBox]):
    bboxes = [bbox_obj for bbox_obj in bounding_boxes if bbox_obj.label == 'mouth']
    
    if len(bboxes) == 0:
        raise IndexError(f"No bounding box found associated with label 'mouth'")

    if len(bboxes) > 1:
        raise ValueError(f"More than one bounding box found with label 'mouth'")

    return bboxes[0]


# integrate this into class itself?
def bbox_intersection(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    # polygon = Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])
    # (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)
    # other_polygon = Polygon([(1, 1), (4, 1), (4, 3.5), (1, 3.5)])
    # (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)
    union = bbox1.area + bbox2.area
    print(f"bbox1: {bbox1.area}, bbox2: {bbox2.area}, union: {union}")

    bbox1_poly = Polgyon([
        (bbox1.xmin, bbox1.ymin),
        (bbox1.xmax, bbox1.ymin),
        (bbox1.xmax, bbox1.ymax),
        (bbox1.xmin, bbox1.ymax)
    ])
    bbox2_poly = Polgyon([
        (bbox2.xmin, bbox2.ymin),
        (bbox2.xmax, bbox2.ymin),
        (bbox2.xmax, bbox2.ymax),
        (bbox2.xmin, bbox2.ymax)
    ])
    intersection = bbox1_poly.intersection(bbox2_poly)

    # intersection = np.lVkkkkkk
    # union = np.logical_or(bbox1, bbox2)
    # iou_score = np.sum(intersection) / np.sum(union)
    # print(f"IoU is: {iou_score}")

