import torch
import torch.nn as nn
import torch.nn.functional as func
from torchvision.models import resnet50
from event_pose_estimation.SMPL import SMPL, batch_rodrigues
from event_pose_estimation.geometry import projection_torch, rot6d_to_rotmat, delta_rotmat_to_rotmat
import numpy as np


class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True
    ):
        super(TemporalEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers,
            batch_first=True
        )

        self.h0_size = n_layers * (bidirectional+1)
        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size*2, 2048)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, 2048)
        self.use_residual = use_residual

    def forward(self, x, hidden_feats):
        # x: [B, T, F]
        # hidden_feats: [B, 2048]
        h0 = hidden_feats.unsqueeze(0).repeat(self.h0_size, 1, 1)
        B, T, F = x.shape
        y, _ = self.gru(x, h0)  # y: [B, T, F]
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(B, T, F)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x
        return y


class Regressor(nn.Module):
    def __init__(self, pose_dim=24*6):
        super(Regressor, self).__init__()
        # self.fc1 = nn.Linear(2048, 1024)
        # self.decpose = nn.Linear(1024, pose_dim)
        # self.dectrans = nn.Linear(1024, 3)
        self.decpose = nn.Linear(2048, pose_dim)
        self.dectrans = nn.Linear(2048, 3)


    def forward(self, x):
        # x: [B, T, 2048]
        B, T, F = x.size()
        x = x.view(-1, F)
        # x = func.dropout(self.fc1(x))
        x1 = self.decpose(x)  # [B*T, pose_dim]
        x2 = self.dectrans(x)  # [B*T, 3]
        delta_pose = x1.view(B, T, x1.size(-1))
        delta_tran = x2.view(B, T, x2.size(-1))
        return delta_pose, delta_tran


class VIBERegressor(nn.Module):
    def __init__(self, pose_dim=24*6, smpl_mean_params='../smpl_model/events_smpl_mean_params.npy'):
        super(VIBERegressor, self).__init__()
        self.fc1 = nn.Linear(2048 + pose_dim + 3, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, pose_dim)
        self.dectrans = nn.Linear(1024, 3)

        mean_params = np.load(smpl_mean_params, allow_pickle=True).item()
        init_pose = torch.from_numpy(mean_params['pose']).float().unsqueeze(0)
        # init_trans = torch.from_numpy(mean_params['trans']).float().unsqueeze(0)
        init_trans = torch.zeros([1, 3]).float()
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_trans', init_trans)

    def forward(self, x, n_iter=3):
        # x: [B, T, 2048]
        B, T, F = x.size()
        x = x.view(-1, F)

        init_pose = self.init_pose.expand(B*T, -1)
        init_trans = self.init_trans.expand(B*T, -1)

        pred_pose = init_pose
        pred_trans = init_trans
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_trans], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose  # [B*T, pose_dim]
            pred_trans = self.dectrans(xc) + pred_trans  # [B*T, 3]

        delta_pose = pred_pose.view(B, T, pred_pose.size(-1))
        delta_tran = pred_trans.view(B, T, pred_trans.size(-1))
        return delta_pose, delta_tran


class EventTrackNet(nn.Module):
    def __init__(
            self,
            events_input_channel=8,
            smpl_dir='../smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl',
            batch_size=8,
            num_steps=8,
            n_layers=1,
            hidden_size=2048,
            bidirectional=False,
            add_linear=False,
            use_residual=True,
            pose_dim=24*6,
            use_flow=True,
            vibe_regressor=False,
            cam_intr=None,
            smpl_mean_params='../smpl_model/events_smpl_mean_params.npy'
    ):
        super(EventTrackNet, self).__init__()
        self.cam_intr = cam_intr
        self.num_steps = num_steps
        self.img_encoder = self.resnet50_encoder(events_input_channel+2*use_flow)  # 2048

        self.tmp_encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )

        self.vibe_regressor = vibe_regressor
        if self.vibe_regressor:
            self.regressor = VIBERegressor(pose_dim, smpl_mean_params)
        else:
            self.regressor = Regressor(pose_dim)
        self.smpl = SMPL(smpl_dir, batch_size * num_steps)

    def resnet50_encoder(self, input_channel):
        model = resnet50(pretrained=False)
        model.conv1 = torch.nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Sequential()
        return model

    def forward(self, events, init_shape, pred_flow=None, hidden_feats=None):
        # events, pred_flow: [B, T, C, H, W]
        # init_shape: [B, 1, 85]
        # hidden_feats: [B, 2048]
        B, T, C, H, W = events.size()

        if pred_flow is not None:
            # [B*T, 8+2/3, H, W]
            x = torch.cat([events, pred_flow], dim=2).view(-1, events.size(2)+pred_flow.size(2), H, W)
        else:
            x = events.view(-1, events.size(2), H, W)
        x = self.img_encoder(x).view(B, T, 2048)
        x = self.tmp_encoder(x, hidden_feats).contiguous()  # [B, T, 2048]

        # if self.vibe_regressor:
        if False:
            pose, trans = self.regressor(x)  # pose [B, T, 24*6], tran [B, T, 3]
            trans = trans.unsqueeze(dim=2)  # [B, T, 1, 3]
            pred_rotmats = rot6d_to_rotmat(pose).view(B, T, 24, 3, 3)
            init_rotmats = batch_rodrigues(init_shape[:, 0, 3:75].reshape(-1, 3)).view(B, 24, 3, 3)  # [B, 24, 3, 3]
            delta_tran = torch.zeros_like(trans)
        else:
            delta_pose, delta_tran = self.regressor(x)  # pose [B, T, 24*6], tran [B, T, 3]

            # trans for each step
            trans = init_shape[:, :, 0:3] + torch.cumsum(delta_tran, dim=1)
            # trans = init_shape[:, :, 0:3] + delta_tran
            trans = trans.unsqueeze(dim=2)  # [B, T, 1, 3]

            # delta_rotmats [B, T, 24, 3, 3]
            delta_rotmats = rot6d_to_rotmat(delta_pose).view(B, T, 24, 3, 3)
            init_rotmats = batch_rodrigues(init_shape[:, 0, 3:75].reshape(-1, 3)).view(B, 24, 3, 3)  # [B, 24, 3, 3]
            pred_rotmats = delta_rotmat_to_rotmat(init_rotmats, delta_rotmats, T)  # [B, T, 24, 3, 3]
            # pred_rotmats = torch.matmul(delta_rotmats, init_rotmats.unsqueeze(1))  # [B, T, 24, 3, 3]

        beta = init_shape[:, :, 75:85].repeat(1, T, 1)  # [B, T, 10]
        verts, joints3d, _ = self.smpl(beta=beta.view(-1, 10),
                                       theta=None,
                                       get_skin=True,
                                       rotmats=pred_rotmats.view(-1, 24, 3, 3))
        verts = verts.view(B, T, verts.size(1), verts.size(2)) + trans.detach()

        init_verts, _, _ = self.smpl(beta=init_shape[:, 0, 75:85],
                                     theta=None,
                                     get_skin=True,
                                     rotmats=init_rotmats)
        init_verts = (init_verts + init_shape[:, :, 0:3]).unsqueeze(1)

        results = {}
        results['delta_tran'] = delta_tran
        results['pred_rotmats'] = pred_rotmats  # [B, T, 24, 3, 3]
        results['tran'] = trans  # [B, T, 1, 3]
        results['verts'] = torch.cat([init_verts, verts], dim=1)  # [B, T+1, 6890, 3]
        results['joints3d'] = joints3d.view(B, T, joints3d.size(1), joints3d.size(2)) + trans.detach()  # [B, T, 24, 3]
        if self.cam_intr is not None:
            results['joints2d'] = projection_torch(results['joints3d'], self.cam_intr, H, W)  # [B, T, 24, 2]
            results['cam_intr'] = self.cam_intr
        return results


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:2" if use_cuda else "cpu")

    max_steps = 8
    batch_size = 2
    model = EventTrackNet(
        events_input_channel=8,
        smpl_dir='../smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl',
        batch_size=batch_size,
        num_steps=max_steps,
        n_layers=1,
        hidden_size=2048,
        bidirectional=False,
        add_linear=False,
        use_residual=True,
        pose_dim=24 * 6,
        cam_intr=torch.tensor([1679.3, 1679.3, 641, 641]) * 256 / 1280.
    )
    model = model.to(device=device)
    print('set up model...')
    _x = torch.rand([batch_size, max_steps, 8, 256, 256]).to(device=device, dtype=torch.float32)
    _pred_flow = torch.rand([batch_size, max_steps, 2, 256, 256]).to(device=device, dtype=torch.float32)
    _init_shape = torch.rand([batch_size, 1, 85]).to(device=device, dtype=torch.float32)
    _hidden_feats = torch.rand([batch_size, 2048]).to(device=device, dtype=torch.float32)
    output = model(_x, _init_shape, _pred_flow, _hidden_feats)

    for k, v in output.items():
        print(k, v.size())




