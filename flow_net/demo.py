import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from flow_net.model import OpticalFlowNet
from flow_net.flowlib import flow_to_image


def test(args):
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % args.gpu_id if use_cuda else "cpu")
    dtype = torch.float32

    # set model
    model = OpticalFlowNet(input_channel=args.input_channel, output_channel=2, num_layers=4, base_channel=32)
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    model_dir = '%s/model/model_flow_%s_%i.pkl' % (args.result_dir, args.source, args.img_size)
    model.load_state_dict(torch.load(model_dir, map_location=device))

    # load data
    events = []
    for i in range(args.skip):
        events.append(cv2.imread('%s/events_%i/%s/event%04i.png' %
                                 (args.data_dir, args.img_size, args.action, args.frame_idx + i), -1))
    events = np.concatenate(events, axis=2)

    fullpic1 = cv2.imread('%s/full_pic_%i/%s/fullpic%04i.jpg' %
                          (args.data_dir, args.img_size, args.action, args.frame_idx))[:, :, 0]
    fullpic2 = cv2.imread('%s/full_pic_%i/%s/fullpic%04i.jpg' %
                          (args.data_dir, args.img_size, args.action, args.frame_idx + args.skip))[:, :, 0]

    events = torch.from_numpy(np.transpose(events, [2, 0, 1])).float().unsqueeze(dim=0)
    dt = torch.tensor([1]).float()

    events_frame = events.to(device=device, dtype=dtype)
    dt = dt.to(device=device, dtype=dtype).view(events_frame.size()[0], 1, 1, 1)
    print(events_frame.size(), dt.size())

    multi_scale_flows = model(events_frame, dt)
    pred_flow = multi_scale_flows[-1].detach().cpu().numpy()[0]  # [2, H, W]
    print(np.max(pred_flow))

    plt.figure(figsize=(5, 5 * 4))
    plt.subplot(4, 1, 1)
    plt.imshow(fullpic1, cmap='gray')
    plt.axis('off')

    flow_rgb = flow_to_image(np.transpose(pred_flow, [1, 2, 0]))
    plt.subplot(4, 1, 2)
    plt.imshow(flow_rgb)
    plt.axis('off')

    flow_int = ((pred_flow * 100).astype(np.int16)).astype(np.float32) / 100
    print(np.max(np.abs(pred_flow-flow_int)))
    flow_int_rgb = flow_to_image(np.transpose(flow_int, [1, 2, 0]))
    plt.subplot(4, 1, 3)
    plt.imshow(flow_int_rgb, cmap='gray')
    plt.axis('off')

    plt.subplot(4, 1, 4)
    plt.imshow(fullpic2, cmap='gray')
    plt.axis('off')
    plt.show()




def get_args():
    def print_args(args):
        """ Prints the argparse argmuments applied
        Args:
          args = parser.parse_args()
        """
        _args = vars(args)
        max_length = max([len(k) for k, _ in _args.items()])
        for k, v in _args.items():
            print(' ' * (max_length - len(k)) + k + ': ' + str(v))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='3')
    parser.add_argument('--data_dir', type=str, default='/home/shihao/data_event')
    parser.add_argument('--result_dir', type=str, default='/home/shihao/data_event')

    parser.add_argument('--input_channel', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_skip', type=int, default=4)
    parser.add_argument('--skip', type=int, default=2)
    parser.add_argument('--max_num_events', type=int, default=60000)
    parser.add_argument('--source', type=str, default='events')

    parser.add_argument('--action', type=str, default='subject05_group2_time3')
    parser.add_argument('--frame_idx', type=int, default=606)

    args = parser.parse_args()
    print_args(args)
    return args


def main():
    args = get_args()
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    test(args)


if __name__ == '__main__':
    main()
