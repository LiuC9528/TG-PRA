'''
 * The Inference of RAM and Tag2Text Models
 * Written by Xinyu Huang
'''
import torch

def inference_tag(image, model):
    
    with torch.no_grad():
        # image_feature, tags, tags_chinese = model.generate_tag(image)
        tags, tags_chinese = model.generate_tag(image)
    
    # return image_feature, [tdags[0],tags_chinese[0]]
    return tags[0],tags_chinese[0]
