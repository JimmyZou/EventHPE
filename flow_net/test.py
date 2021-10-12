import torch
from torch.utils.data import DataLoader
import os
import time
import joblib
import numpy as np
import sys
sys.path.append('../')
from flow_net.model import OpticalFlowNet
from flow_net.dataloader import FlowDataloader


def test(args):
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % args.gpu_id if use_cuda else "cpu")
    dtype = torch.float32

    # set dataset
    dataset_val = FlowDataloader(data_dir=args.data_dir, input_channel=args.input_channel,
                                 num_skip=args.num_skip, skip=args.skip, max_num_events=args.max_num_events,
                                 transform=None, mode='test', img_size=args.img_size, source=args.source)
    val_generator = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=False)

    # '''
    # set model
    model = OpticalFlowNet(input_channel=args.input_channel, output_channel=2, num_layers=4, base_channel=32)
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    model_dir = '%s/model/model_flow_%s_%i.pkl' % (args.result_dir, args.source, args.img_size)
    model.load_state_dict(torch.load(model_dir, map_location=device))

    # check path
    save_dir = '%s/pred_flow_%s_%i' % (args.result_dir, args.source, args.img_size)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # validation
    print('------------------------------------- test ------------------------------------')
    start_time = time.time()
    model.eval()  # dropout layers will not work in eval mode
    with torch.set_grad_enabled(False):  # deactivate autograd to reduce memory usage
        for idx, batch_data in enumerate(val_generator):
            # batch_data: {img1, img2, events_frame, dt}
            img1 = batch_data['img1'].to(device=device, dtype=dtype)
            img2 = batch_data['img2'].to(device=device, dtype=dtype)
            events_frame = batch_data['events_frame'].to(device=device, dtype=dtype)
            dt = batch_data['dt'].to(device=device, dtype=dtype).view(img1.size()[0], 1, 1, 1)

            if args.source == 'events':
                multi_scale_flows = model(events_frame, dt)
            elif args.source == 'full_pic':
                multi_scale_flows = model(torch.cat([img1, img2], dim=1) / 255., dt)
            else:
                multi_scale_flows = model(torch.cat([img1, img2], dim=1) / 255., dt)

            pred_flow = multi_scale_flows[-1].cpu().numpy()
            for i, info in enumerate(batch_data['info']):
                action, frame_idx = info.split('-')
                # print(action, frame_idx)
                if not os.path.exists('%s/%s' % (save_dir, action)):
                    os.mkdir('%s/%s' % (save_dir, action))

                # import matplotlib.pyplot as plt
                # from flow_net.flowlib import flow_to_image
                # flow_rgb = flow_to_image(np.transpose(pred_flow[i, :, :, :], [1, 2, 0]))
                # plt.figure()
                # plt.imshow(flow_rgb)
                # plt.show()

                fname = '%s/%s/flow%s.pkl' % (save_dir, action, frame_idx)
                flow_int = (pred_flow[i, :, :, :] * 100).astype(np.int16)
                joblib.dump(flow_int, fname, compress=3)

            # break
        end_time = time.time()
        time_used = (end_time - start_time) / 60.
        print('time used: %.2f' % time_used)
    # '''


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
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', type=str, default='/home/shihao/data_event')
    parser.add_argument('--result_dir', type=str, default='/home/shihao/data_event')

    parser.add_argument('--input_channel', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--smoothness_loss_weight', type=float, default=0.5)
    parser.add_argument('--num_skip', type=int, default=1)
    parser.add_argument('--skip', type=int, default=2)
    parser.add_argument('--max_num_events', type=int, default=60000)
    parser.add_argument('--source', type=str, default='events')

    parser.add_argument('--batch_size', '-bs', type=int, default=8)
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
