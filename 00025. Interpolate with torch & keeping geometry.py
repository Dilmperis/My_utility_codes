import torch.nn.functional as F

#To preserve object geometry (pixel alignment, coordinates, and scaling) when resizing in 2D:
F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=True)

# or in 3D:
F.interpolate(x, scale_factor=(sx, sy, sz), mode='trilinear', align_corners=True)

