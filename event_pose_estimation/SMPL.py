'''
    file:   SMPL.py

    date:   2018_05_03
    author: zhangxiong(1025679612@qq.com)
    mark:   the algorithm is cited from original SMPL
'''
import torch
import pickle
import sys
sys.path.append('../')
import numpy as np
import torch.nn as nn
from plyfile import PlyData, PlyElement
import cv2
import torch.nn.functional as F


"""SMPL pytorch implementation"""


def batch_global_rigid_transformation(Rs, Js, parent, device):
    N = Rs.shape[0]
    root_rotation = Rs[:, 0, :, :]
    Js = torch.unsqueeze(Js, -1)  # [N, 24, 3, 1]

    def make_A(R, t):
        R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])  # [N, 4, 3]
        t_homo = torch.cat([t, torch.ones((N, 1, 1), device=device)], dim=1)  # [N, 4, 1]
        return torch.cat([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])  # [N, 4, 4]
    results = [A0]

    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]  # [N, 3, 1]
        A_here = make_A(Rs[:, i], j_here)  # [N, 4, 4]
        res_here = torch.matmul(results[parent[i]], A_here)  # [N, 4, 4] H_{parent} * H_{here}
        results.append(res_here)

    results = torch.stack(results, dim=1)  # [N, 24, 4, 4]

    new_J = results[:, :, :3, 3]  # [N, 24, 3]
    Js_w0 = torch.cat([Js, torch.zeros((N, 24, 1, 1), device=device)], dim=2)  # [N, 24, 4, 1]
    init_bone = torch.matmul(results, Js_w0)  # [N, 24, 4, 1]
    init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])  # [N, 24, 4, 4]
    A = results - init_bone

    return new_J, A


def batch_rodrigues(theta):
    # theta N x 3
    # batch_size = theta.shape[0]
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)

    return quat2mat(quat)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


class SMPL(nn.Module):
    def __init__(self, model_path, max_batch_size):
        super(SMPL, self).__init__()

        self.model_path = model_path
        model = pickle.load(open(model_path, 'rb'), encoding='iso-8859-1')
        self.faces = model['f']

        np_v_template = np.array(model['v_template'], dtype=np.float)
        self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
        self.size = [np_v_template.shape[0], 3]

        np_shapedirs = np.array(model['shapedirs'], dtype=np.float)
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())  # [10, 20670]

        np_J_regressor = np.array(model['J_regressor'].todense().T, dtype=np.float)
        self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())  # [6890, 24]

        np_posedirs = np.array(model['posedirs'], dtype=np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())  # [207, 20670]

        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)

        np_weights = np.array(model['weights'], dtype=np.float)  # [6890, 24]
        vertex_count = np_weights.shape[0]
        vertex_component = np_weights.shape[1]

        np_weights = np.tile(np_weights, (max_batch_size, 1))  # [N, 6890, 24]
        self.register_buffer('weight', torch.from_numpy(np_weights).float().reshape(-1, vertex_count, vertex_component))
        self.register_buffer('e3', torch.eye(3).float())
        self.cur_device = None

    def save_obj(self, verts, obj_mesh_name):
        vertex = np.array([tuple(i) for i in verts], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        face = np.array([(tuple(i), 255, 255, 255) for i in self.faces],
                        dtype=[('vertex_indices', 'i4', (3,)), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        el = PlyElement.describe(vertex, 'vertex')
        el2 = PlyElement.describe(face, 'face')
        plydata = PlyData([el, el2])
        plydata.write(obj_mesh_name)

    def get_root_rt(self, beta, theta):
        if not self.cur_device:
            device = beta.device
            self.cur_device = torch.device(device.type, device.index)

        # [N, 6890, 3]
        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)  # [N, 24, 3]

        t = J[:, 0, :]  # [N, 3]
        R = batch_rodrigues(theta.view(-1, 24, 3)[:, 0, :])  # [N, 3, 3]
        return R, t

    def forward(self, beta, theta=None, get_skin=False, rotmats=None):
        if not self.cur_device:
            device = beta.device
            self.cur_device = torch.device(device.type, device.index)

        num_batch = beta.size(0)

        # [N, 6890, 3]
        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)  # [N, 24, 3]

        if rotmats is not None:
            Rs = rotmats
        else:
            Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = torch.sub(Rs[:, 1:, :, :], self.e3, alpha=1.0).view(-1, 207)
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, self.cur_device)

        weight = self.weight[:num_batch]  # [N, 6890, 24]
        W = weight.view(num_batch, -1, 24)
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)  # [N, 6890, 4, 4]

        v_posed_homo = torch.cat(
            [v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device=self.cur_device)], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]

        joints = self.J_transformed

        if get_skin:
            return verts, joints, Rs
        else:
            return joints


