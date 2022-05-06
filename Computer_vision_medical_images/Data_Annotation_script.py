from PIL import Image
import numpy as np                         
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
import json
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import gc

def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]

            # If the pixel is not white...(we have white background)
            if pixel != (255, 255, 255):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks
    
def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    #     measure.find_contours
    #Uses the “marching squares” method to compute a the iso-valued contours of 
    #the input 2D array for a particular level value. 
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')
    bboxes = []
    segmentations = []
    polygons = []
    areas = []
    annotation = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        #shapely.geometry Polygon
        
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)
    # Combine the polygons to calculate the bounding box and area
    offset = 0
    steps = 0
    for i , polygon in enumerate(polygons):
        
        bbox = []
        area = []
        if not polygon.is_empty:
            bbox = polygon.bounds
            area = polygon.area
        annotation = {
        'segmentation': segmentations[i],
        'iscrowd': 0,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area}
        annotations.append(annotation)
        steps +=1
        annotation_id += 1
    return annotations , steps

data_path = ".\data\lesion"
paths=[]
for (root,dirs,files) in os.walk(data_path, topdown=True):
    for filename in files:
        if filename.endswith('.bmp'):
            path = os.path.join(root, filename)
            paths.append(path)
lesion_id = 1
is_crowd= 0
image_id = 1
annotations = []
annotation_id = 1
for mask_image_path in tqdm(paths):
    gc.collect()
    with Image.open(mask_image_path) as mask_image:
        path_split = mask_image.filename.split('\\')
        image_id = path_split[3]+"."+path_split[4].split('.')[0]
        print(image_id)
        sub_masks = create_sub_masks(mask_image)
        for color, sub_mask in sub_masks.items():
            category_id = lesion_id
            annotation, steps = create_sub_mask_annotation(sub_mask, image_id, category_id,annotation_id, is_crowd)
            annotations=annotations+annotation
            annotation_id += steps
with open('lesion_annotations.json','w') as outfile:
    json.dump(annotations,outfile)