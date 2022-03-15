"""
usage
# add for roi pooling
import torch
import torchvision.ops.roi_align as roi_align
self.roi_align = roi_align
"""

def region_embed(self, x, bbox):
    """
    Args:
        x (): the input video
        bbox (): bounding boxes with 4 loc + height/width; stacked for num_frame times

    Returns:
        the raw pixel region of bbox
    """
    b, t, c, h, w = x.size()
    x = x.view(-1, c, h, w)
    B, L, N = bbox.size()
    coordinates = torch.zeros((B * L, 5)).cuda()
    for i in range(B * L):
        coordinates[i][0] = i // L
        coordinates[i][1:] = bbox[i // L, i % L, :4]
    regions = self.roi_align(x, coordinates, output_size=[self.patch_size, self.patch_size])
    region_features = self.region_embedding_layer(regions)
    region_features = region_features.view(-1, L // t, self.embed_dim)
    return region_features