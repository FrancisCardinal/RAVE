import torch
import torch.nn as nn

from models.common import Conv, NMS
from utils.torch_utils import fuse_conv_and_bn, model_info


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        # self.no = nc + 5  # number of outputs per anchor
        self.no = nc + 5 + 10  # number of outputs per anchor

        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer(
            "anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2)
        )  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch
        )  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        # self.training |= self.export
        if self.export:
            for i in range(self.nl):
                x[i] = self.m[i](x[i])
                bs, _, ny, nx = x[
                    i
                ].shape  # x(bs,48,20,20) to x(bs,3,20,20,16)
                x[i] = (
                    x[i]
                    .view(bs, self.na, self.no, ny, nx)
                    .permute(0, 1, 3, 4, 2)
                    .contiguous()
                )

            return x
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = torch.full_like(x[i], 0)
                class_range = list(range(5)) + list(range(15, 15 + self.nc))
                y[..., class_range] = x[i][..., class_range].sigmoid()
                y[..., 5:15] = x[i][..., 5:15]
                # y = x[i].sigmoid()

                y[..., 0:2] = (
                    y[..., 0:2] * 2.0 - 0.5 + self.grid[i].to(x[i].device)
                ) * self.stride[
                    i
                ]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[
                    i
                ]  # wh

                # y[..., 5:15] = y[..., 5:15] * 8 - 4
                y[..., 5:7] = (
                    y[..., 5:7] * self.anchor_grid[i]
                    + self.grid[i].to(x[i].device) * self.stride[i]
                )  # landmark x1 y1
                y[..., 7:9] = (
                    y[..., 7:9] * self.anchor_grid[i]
                    + self.grid[i].to(x[i].device) * self.stride[i]
                )  # landmark x2 y2
                y[..., 9:11] = (
                    y[..., 9:11] * self.anchor_grid[i]
                    + self.grid[i].to(x[i].device) * self.stride[i]
                )  # landmark x3 y3
                y[..., 11:13] = (
                    y[..., 11:13] * self.anchor_grid[i]
                    + self.grid[i].to(x[i].device) * self.stride[i]
                )  # landmark x4 y4
                y[..., 13:15] = (
                    y[..., 13:15] * self.anchor_grid[i]
                    + self.grid[i].to(x[i].device) * self.stride[i]
                )  # landmark x5 y5

                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid(
            [torch.arange(ny), torch.arange(nx)], indexing="ij"
        )
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(
        self, cfg="yolov5s.yaml", ch=3, nc=None
    ):  # model, input channels, number of classes
        super(Model, self).__init__()

    def forward(self, x):
        y = []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        return x

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print("Fusing layers... ")
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    # add or remove NMS module
    def nms(self, mode=True, conf=0.25, iou=0.45, classes=None):
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print("Adding NMS... ")
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name="%s" % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print("Removing NMS... ")
            self.model = self.model[:-1]  # remove
        return self

    def info(self, verbose=False):  # print model information
        model_info(self, verbose)
