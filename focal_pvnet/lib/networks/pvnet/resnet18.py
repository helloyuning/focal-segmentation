from torch import nn
import torch
from torch.nn import functional as F
# from .resnet import resnet18
from lib.networks.pvnet.resnet import resnet18
from lib.csrc.ransac_voting.ransac_voting_gpu import ransac_voting_layer, ransac_voting_layer_v3, estimate_voting_distribution_with_mean
from lib.config import cfg
from lib.networks.pvnet.Det import detnet59
# from mit_semseg.models.hrnet import hrnetv2
from mit_semseg.models.resnext import resnext101

class Resnet18(nn.Module):
    def __init__(self, ver_dim, seg_dim, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32):
        super(Resnet18, self).__init__()
        self.embed_size = 300
        self.hidden_size = 512
        self.num_layer = 1
        #self.lstm = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=num_layer,batch_first=True)
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet18(fully_conv=True,
                               pretrained=False,
                               output_stride=8,
                               remove_avg_pool_layer=True)
        # resnet18_8s = detnet59(fully_conv=True,
        #                        pretrained=False,
        #                        output_stride=8,
        #                        remove_avg_pool_layer=True)


        self.ver_dim=ver_dim
        self.seg_dim=seg_dim

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True)
        )


        self.resnet18_8s = resnet18_8s

        # x8s->128
        self.conv8s=nn.Sequential(
            nn.Conv2d(128+fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1,True)
        )
        # self.conv8s = nn.Sequential(#fcdim=256, 源代码卷积和加
        #     nn.Conv2d(512 + fcdim, s8dim, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(s8dim),
        #     nn.LeakyReLU(0.1, True)
        # )

        self.up8sto4s=nn.UpsamplingBilinear2d(scale_factor=2)
        # x4s->64
        self.conv4s=nn.Sequential(
            nn.Conv2d(64+s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1,True)
        )
        #print("s8dim",s8dim)
        # self.conv4s = nn.Sequential(#s8dim=128
        #     nn.Conv2d(256 + s8dim, s4dim, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(s4dim),
        #     nn.LeakyReLU(0.1, True)
        # )


        # x2s->64
        self.conv2s=nn.Sequential(
            nn.Conv2d(64+s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1,True)
        )

        self.up4sto2s=nn.UpsamplingBilinear2d(scale_factor=2)

        self.convraw = nn.Sequential(
            nn.Conv2d(3+s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(raw_dim, seg_dim+ver_dim, 1, 1)
        )
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def decode_keypoint(self, output):
        vertex = output['vertex'].permute(0, 2, 3, 1)
        b, h, w, vn_2 = vertex.shape
        vertex = vertex.view(b, h, w, vn_2//2, 2)
        mask = torch.argmax(output['seg'], 1)

        if cfg.test.un_pnp:
            mean = ransac_voting_layer_v3(mask, vertex, 512, inlier_thresh=0.99)
            kpt_2d, var = estimate_voting_distribution_with_mean(mask, vertex, mean)

            output.update({'mask': mask, 'kpt_2d': kpt_2d, 'var': var})
        else:
            kpt_2d = ransac_voting_layer_v3(mask, vertex, 128, inlier_thresh=0.99, max_num=100)

            output.update({'mask': mask, 'kpt_2d': kpt_2d})

    def forward(self, x, feature_alignment=False):
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x)
        # x2s, x4s, x8s, xfc = self.resnet18_8s(x)
        # print("过了")
        fm=self.conv8s(torch.cat([xfc,x8s],1))
        fm=self.up8sto4s(fm)
        if fm.shape[2]==136:
            fm = nn.functional.interpolate(fm, (135,180), mode='bilinear', align_corners=False)

        fm=self.conv4s(torch.cat([fm,x4s],1))
        fm=self.up4sto2s(fm)

        fm=self.conv2s(torch.cat([fm,x2s],1))
        fm=self.up2storaw(fm)

        x=self.convraw(torch.cat([fm,x],1))



        seg_pred=x[:,:self.seg_dim,:,:]
        ver_pred=x[:,self.seg_dim:,:,:]
        # ########

        # bs, ts, h, w = seg_pred.size()
        # bs2, ts2, h2, w2 = ver_pred.size()
        # seg_in = seg_pred.view(bs, ts, -1)
        # ver_in = ver_pred.view(bs2, ts2, -1)
        #
        #
        # if torch.cuda.is_available():
        #     RNN_seg = nn.RNN(input_size=seg_in.shape[2], hidden_size=h, num_layers=1, batch_first=True).cuda()
        #     # lstm_seg = nn.LSTM(input_size=seg_in.shape[2], hidden_size=h, num_layers=1, batch_first=True).cuda()
        #     seg_in = seg_in.cuda()
        #     # seg_out, _ = lstm_seg(seg_in)#可用
        #     seg_out, _ = RNN_seg(seg_in)
        #     d_seg = seg_out.unsqueeze(-1)
        #     r_seg = torch.cat((seg_pred,d_seg),-1)
        #
        #     #lstm_ver = nn.LSTM(input_size=ver_in.shape[2], hidden_size=h2, num_layers=1, batch_first=True).cuda()
        #     RNN_ver = nn.RNN(input_size=ver_in.shape[2], hidden_size=h2, num_layers=1, batch_first=True).cuda()
        #     ver_in = ver_in.cuda()
        #     # ver_out, _ = lstm_ver(ver_in)
        #     ver_out, _ = RNN_ver(ver_in)
        #     d_ver = ver_out.unsqueeze(-1)
        #     r_ver = torch.cat((ver_pred, d_ver), -1)
        #
        #
        #     r_seg = r_seg[:, :, :,torch.arange(r_seg.size(3)) != w]
        #     r_ver = r_ver[:, :, :, torch.arange(r_ver.size(3)) != w2]
        #

        #######
        # ret = {'seg': r_seg, 'vertex': r_ver}
        ret = {'seg': seg_pred, 'vertex': ver_pred}

        if not self.training:
            with torch.no_grad():
                self.decode_keypoint(ret)

        return ret


def get_res_pvnet(ver_dim, seg_dim):
    model = Resnet18(ver_dim, seg_dim)
    return model

