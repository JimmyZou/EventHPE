import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EncoderBlock, self).__init__()
        self.conv_block = nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(),

            torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(),
        )

    def forward(self, _input):
        out = self.conv_block(_input)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DecoderBlock, self).__init__()
        self.conv_block = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, padding=1,
                                     output_padding=1, stride=2, bias=False),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(),

            torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(),
        )

    def forward(self, _input):
        out = self.conv_block(_input)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(),

            torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(),
        )

    def forward(self, _input):
        out = self.conv_block(_input) + _input
        return out


class FlowBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FlowBlock, self).__init__()
        self.conv_block = nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, stride=1, bias=False),
            torch.nn.Tanh(),
        )

    def forward(self, _input):
        out = self.conv_block(_input)
        return out


class OpticalFlowNet(nn.Module):
    def __init__(self,  input_channel=2, output_channel=2, num_layers=4, base_channel=32):
        super(OpticalFlowNet, self).__init__()
        self.encoders = nn.ModuleList()
        in_c, out_c = input_channel, base_channel
        for i in range(num_layers):
            # print(in_c, out_c)
            self.encoders.append(EncoderBlock(in_c, out_c))
            in_c, out_c = out_c, out_c * 2

        self.residual = ResidualBlock(in_c, in_c)

        self.decoders, self.flows = nn.ModuleList(), nn.ModuleList()
        in_c, out_c = out_c, int(in_c / 2)  # 512, 128
        for i in range(num_layers):
            # print(in_c, out_c)
            self.decoders.append(DecoderBlock(in_c + 2 * int(i > 0), out_c))
            self.flows.append(FlowBlock(out_c, output_channel))
            in_c, out_c = int(in_c / 2), int(out_c / 2)

    def forward(self, x, dt):
        img_size = x.size()[2]
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)

        x = self.residual(x)

        flow_pyramid = []
        for i, decoder in enumerate(self.decoders):
            # concatenate skip connections
            x = torch.cat([x, skip_connections[-i-1]], dim=1)
            x = decoder(x)

            pred_flow = self.flows[i](x)
            flow_pyramid.append(pred_flow * img_size / dt)
            x = torch.cat([x, pred_flow], dim=1)
        return flow_pyramid


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:3" if use_cuda else "cpu")
    model = OpticalFlowNet(2, 2, 4, 32)
    model = model.to(device=device)
    _input = torch.rand([2, 2, 512, 512]).to(device=device, dtype=torch.float32)
    _dt = torch.randint(1, 5, [2]).to(device=device, dtype=torch.float32).view(2,1,1,1)
    print(_dt.size())
    output = model(_input, _dt)

