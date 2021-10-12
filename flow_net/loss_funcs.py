import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torch.nn as nn

"""
Robust Charbonnier loss.
"""
def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3):
    loss = torch.mean(torch.pow(torch.mul(delta,delta) + torch.mul(epsilon,epsilon), alpha))
    return loss

"""
warp an image/tensor (im2) back to im1, according to the optical flow
x: [B, C, H, W] (im2), flow: [B, 2, H, W] flow
"""
def warp(x, flow):
    if x.is_cuda:
        device = x.get_device()
    else:
        device = torch.device("cpu")
    B, C, H, W = x.size()

    # mesh grid
    v, u = torch.meshgrid([torch.arange(0, W, device=device), torch.arange(0, H, device=device)])
    grid = torch.stack([u, v], dim=0).repeat(B, 1, 1, 1).float()  # [B, 2, H, W]
    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / W - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / H - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    return output


def compute_smoothness_loss(flows):
    total_smoothness_loss = 0.
    loss_weight_sum = 0.

    for flow in flows:
        # [B, C, H, W]
        flow_ucrop = flow[:, :, 1:, :]
        flow_dcrop = flow[:, :, :-1, :]
        flow_lcrop = flow[:, :, :, 1:]
        flow_rcrop = flow[:, :, :, :-1]

        flow_ulcrop = flow[:, :, 1:, 1:]
        flow_drcrop = flow[:, :, :-1, :-1]
        flow_dlcrop = flow[:, :, :-1, 1:]
        flow_urcrop = flow[:, :, 1:, :-1]

        smoothness_loss = charbonnier_loss(flow_lcrop - flow_rcrop) + \
                          charbonnier_loss(flow_ucrop - flow_dcrop) + \
                          charbonnier_loss(flow_ulcrop - flow_drcrop) + \
                          charbonnier_loss(flow_dlcrop - flow_urcrop)
        total_smoothness_loss += smoothness_loss / 4.
        loss_weight_sum += 1.
    total_smoothness_loss /= loss_weight_sum
    return total_smoothness_loss


def compute_photometric_loss(multi_scale_flows, img1, img2):
    total_photometric_loss = 0.
    loss_weight_sum = 0.

    for flow in multi_scale_flows:
        B, C, H, W = flow.size()
        pre_img = F.interpolate(img1, (H, W), mode='bilinear')
        cur_img = F.interpolate(img2, (H, W), mode='bilinear')
        warped_pre_img = warp(cur_img, flow)

        photometric_loss = charbonnier_loss(pre_img - warped_pre_img)
        total_photometric_loss += photometric_loss
        loss_weight_sum += 1.
    total_photometric_loss /= loss_weight_sum
    return total_photometric_loss


if __name__ == '__main__':
    img_size = 512
    xx, yy = np.meshgrid(np.arange(img_size), np.arange(img_size))
    grid = np.stack([xx, yy], axis=0)
    # print(grid[:, 0:10, 0:10])

    flow = grid - (img_size // 2)
    # flow[:, mask[0], mask[1]] = 0
    mask = np.where(np.linalg.norm(flow, axis=0) > (img_size // 2))

    from flow.flowlib import flow_to_image
    flow_rgb = flow_to_image(np.transpose(-flow, [1, 2, 0]))
    flow_rgb[mask[0], mask[1], :] = 255
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.imshow(flow_rgb)
    plt.axis('off')
    plt.show()

    # x = np.zeros([1, img_size, img_size])
    # center = img_size // 2
    # x[0, center-10:center+10, center-10:center+10] = 1
    #
    # device = torch.device('cpu')
    # x = torch.from_numpy(np.expand_dims(x, axis=0)).float().to(device)
    # flow = torch.from_numpy(np.expand_dims(-flow, axis=0)).float().to(device)
    # warped_img = warp(x, flow)[0]
    # img = warped_img.cpu().numpy()[0]
    # print(img[0:10, 0:10])
