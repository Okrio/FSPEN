
import torch
import torch.nn as nn

class PE(nn.Module):
    def __init__(self, numUnits, width, channel, groups=1) -> None:
        super().__init__()
        self.numUnits = numUnits
        self.width = width
        self.channel = channel
        self.groups = groups
        self.inter_rnn = nn.ModuleList()
        self.inter_fc = nn.ModuleList()
        assert width % groups == 0, "groups is error"
        for i in range(0, groups):
            self.inter_rnn.append(nn.GRU(self.numUnits, self.numUnits, 
                                         batch_first=True))
                              
            self.inter_fc.append(nn.Linear(self.numUnits, self.numUnits))
        self.inter_ln = nn.LayerNorm([self.width, self.numUnits], eps=1e-8)

    def forward(self, x):
        # b t f c
        b,t,f,c = x.shape
        x = x.permute(0,2,1,3)
        x_tmp = torch.chunk(x, self.groups, 1)
        rnn_out = []
        rnn_hidden = []
        for i in range(0, self.groups):
            x_tmp1 = x_tmp[i].reshape(-1,t,c)
            x_tmp1_out, x_tmp1_hidden = self.inter_rnn[i](x_tmp1)
            rnn_hidden.append(x_tmp1_hidden)
            x_fc = self.inter_fc[i](x_tmp1_out)
            x_fc = x_fc.reshape(b,-1,t,self.numUnits)
            rnn_out.append(x_fc)
        
        x_merge = torch.cat(rnn_out, 1) # b f t c
        x_merge = x_merge.permute(0,2,1,3) # b t f c
        x_merge = self.inter_ln(x_merge)

        return x_merge, rnn_hidden
    
class SUBBAND_ENCODE(nn.Module):
    def __init__(self, in_ch=1,out_ch=[32, 32, 32, 32,32], kernel_size=[4, 7, 11, 20, 40], stride_size=[2, 3,5,10,20]) -> None:
        super().__init__()
        # 8 7 6 7 6 || 16 21 30 70 120 
        self.conv1d_1 = nn.Conv1d(in_ch, out_ch[0], kernel_size[0], stride_size[0],padding=1)
        self.conv1d_2 = nn.Conv1d(in_ch, out_ch[1], kernel_size[1], stride_size[1], padding=2)
        self.conv1d_3 = nn.Conv1d(in_ch, out_ch[2], kernel_size[2], stride_size[2], padding=3)
        self.conv1d_4 = nn.Conv1d(in_ch, out_ch[3], kernel_size[3], stride_size[3], padding=5)
        self.conv1d_5 = nn.Conv1d(in_ch, out_ch[4], kernel_size[4], stride_size[4], padding=10)
        self.norm1d_1 = nn.BatchNorm1d(out_ch[0])
        self.norm1d_2 = nn.BatchNorm1d(out_ch[1])
        self.norm1d_3 = nn.BatchNorm1d(out_ch[2])
        self.norm1d_4 = nn.BatchNorm1d(out_ch[3])
        self.norm1d_5 = nn.BatchNorm1d(out_ch[4])
        self.act = nn.PReLU()
    def forward(self, x):
        # b*t c f
        # b,f,t,c = x.shape
        # x = x.permute(0,2,3,1)
        # x = x.reshape(b*t,c,f)
        x1 = self.act(self.norm1d_1(self.conv1d_1(x[...,:16])))
        x2 = self.act(self.norm1d_2(self.conv1d_2(x[...,16:37])))
        x3 = self.act(self.norm1d_3(self.conv1d_3(x[...,37:67])))
        x4 = self.act(self.norm1d_4(self.conv1d_4(x[...,67:137])))
        x5 = self.act(self.norm1d_5(self.conv1d_5(x[...,137:])))

        return x1,x2,x3,x4,x5


class SUBBAND_DECODE(nn.Module):
    def __init__(self, in_ch=[32, 32, 32, 32, 32],out_ch=[2, 3, 5, 10, 20]) -> None:
        super().__init__()
        self.decode_1 = nn.Sequential(nn.Linear(in_ch[0]*2, out_ch[0]), nn.ReLU())
        self.decode_2 = nn.Sequential(nn.Linear(in_ch[1]*2, out_ch[1]), nn.ReLU())
        self.decode_3 = nn.Sequential(nn.Linear(in_ch[2]*2, out_ch[2]), nn.ReLU())
        self.decode_4 = nn.Sequential(nn.Linear(in_ch[3]*2, out_ch[3]), nn.ReLU())
        self.decode_5 = nn.Sequential(nn.Linear(in_ch[4]*2, out_ch[4]), nn.ReLU())

    def forward(self, x, y):
        # b*t, c,f  b*t, c,f
        b = y.shape[0]
        x_1 = torch.cat([x[0], y[...,:8]], 1).permute(0,2,1)
        x_1 = self.decode_1(x_1).reshape(b, -1)

        x_2 = torch.cat([x[1], y[...,8:15]], 1).permute(0,2,1)
        x_2 = self.decode_2(x_2).reshape(b, -1)

        x_3 = torch.cat([x[2], y[...,15:21]], 1).permute(0,2,1)
        x_3 = self.decode_3(x_3).reshape(b, -1)

        x_4 = torch.cat([x[3], y[...,21:28]], 1).permute(0,2,1)
        x_4 = self.decode_4(x_4).reshape(b, -1)

        x_5 = torch.cat([x[4], y[...,28:]], 1).permute(0,2,1)
        x_5 = self.decode_5(x_5).reshape(b, -1)

        out = torch.cat([x_1, x_2, x_3, x_4, x_5], -1)
        
        return out

        



class DPRNN_GRU(nn.Module):
    def __init__(self, numUnits, width, channel) -> None:
        super().__init__()
        self.numUnits = numUnits
        self.width = width
        self.channel = channel
        self.intra_rnn = nn.GRU(input_size=self.numUnits, hidden_size=self.numUnits, batch_first=True, bidirectional=True)
        self.intra_fc = nn.Linear(self.numUnits*2, self.numUnits)
        self.intra_ln = nn.LayerNorm([self.width, self.numUnits], eps=1e-8)

        self.inter_rnn = nn.GRU(input_size=self.numUnits, hidden_size=self.numUnits, batch_first=True)
        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)
        self.inter_ln = nn.LayerNorm([self.width, self.numUnits], eps=1e-8)
        self.inter_rnn1 = PE(numUnits, width, channel, 8)

    def forward(self, x):
        # bctf
        b,c,t,f = x.shape
        x = x.permute(0, 2, 3, 1)
        intra_x = x.reshape(b*t, f, c)
        intra_x = self.intra_rnn(intra_x)[0]
        intra_x = self.intra_fc(intra_x)
        intra_x = intra_x.reshape(b, t, -1, c)
        intra_x = self.intra_ln(intra_x)

        intra_out = x + intra_x

        # x = intra_out.permute(0, 2, 1, 3)
        # inter_x = x.reshape(b*f, t, c)
        # inter_x = self.inter_rnn(inter_x)[0]
        # inter_x = self.inter_fc(inter_x)
        # inter_x = inter_x.reshape(b,f,-1,c)
        # inter_x = inter_x.permute(0,2,1,3)
        # inter_x = self.inter_ln(inter_x)

        inter_x = self.inter_rnn1(intra_out)[0]

        inter_out = intra_x + inter_x

        dual_out = inter_out.permute(0,3,1,2)
        return dual_out
    
    
class DPCRN_Light(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, filter_size=[4,16,32], kernel_size=[6,8,6],stride_size=[2,2,2], numUnits=16):
        super(DPCRN_Light, self).__init__()
        self.filter_size = filter_size
        self.conv1d_1 = nn.Conv1d(in_ch, filter_size[0], kernel_size[0], stride_size[0], padding=[2])
        self.conv1d_2 = nn.Conv1d(filter_size[0], filter_size[1], kernel_size[1],stride_size[1], padding=[3])
        self.conv1d_3 = nn.Conv1d(filter_size[1], filter_size[2], kernel_size[2], stride_size[2], padding=[2])
        self.conv1X1d = nn.Conv1d(filter_size[2], filter_size[2], 1,1)

        self.act = nn.ELU()

        self.norm1d_1 = nn.BatchNorm1d(filter_size[0])
        self.norm1d_2 = nn.BatchNorm1d(filter_size[1])
        self.norm1d_3 = nn.BatchNorm1d(filter_size[2])

        self.deconv1 = nn.Sequential(nn.Conv1d(filter_size[2]*2, filter_size[2], 1, 1),
                                     nn.ConvTranspose1d(filter_size[2], filter_size[1], kernel_size[0], stride_size[0], padding=2))
        self.deconv2 = nn.Sequential(nn.Conv1d(filter_size[1]*2, filter_size[1], 1, 1),
                                     nn.ConvTranspose1d(filter_size[1], filter_size[0], kernel_size[1], stride_size[1], padding=3))
        self.deconv3 = nn.Sequential(nn.Conv1d(filter_size[0]*2, filter_size[0], 1, 1),
                                     nn.ConvTranspose1d(filter_size[0], out_ch, kernel_size[2], stride_size[2], padding=2,output_padding=1))
        self.denorm1d_1 = nn.BatchNorm1d(filter_size[1])
        self.denorm1d_2 = nn.BatchNorm1d(filter_size[0])
        self.denorm1d_3 = nn.BatchNorm1d(out_ch)
        self.dprnn = DPRNN_GRU(numUnits, 32, 16)
        self.dprnn1 = DPRNN_GRU(numUnits, 32, 16)
        self.dprnn2 = DPRNN_GRU(numUnits, 32, 16)

        self.conv1_fm = nn.Conv1d(32, 16, 1, 1)
        self.deconv1_fm = nn.Conv1d(16, 32, 1, 1)
        
    def forward(self, x):
        # bftc 
        x_ref = x
        x = x.permute(0,2,1,3)
        b,t,f,c = x.shape
        x1 = x.reshape(b*t,f,c)
        x1 = x1.permute(0,2,1) # b*t c, f
        x1 = self.conv1d_1(x1)
        x1 = self.act(self.norm1d_1(x1))
        x2 = self.conv1d_2(x1)
        x2 = self.act(self.norm1d_2(x2))
        x3 = self.conv1d_3(x2)
        x3 = self.act(self.norm1d_3(x3))
        x4 = self.conv1X1d(x3)
        x4 = self.conv1_fm(x4)

        x_rnn_in = x4.reshape(b,t,self.filter_size[-1]//2, -1)
        x_rnn_in = x_rnn_in.permute(0, 2, 1, 3) # b c t f
        rnn_out = self.dprnn(x_rnn_in)
        rnn_out = self.dprnn1(rnn_out)
        rnn_out = self.dprnn2(rnn_out)

        rnn_out = rnn_out.permute(0, 2, 1, 3)
        rnn_out = rnn_out.reshape(b*t, self.filter_size[-1]//2, -1) # b*t c f
        rnn_out = self.deconv1_fm(rnn_out)
        o_1 = torch.cat([rnn_out, x3], 1)
        o_1 = self.deconv1(o_1)
        o_1 = self.denorm1d_1(o_1)
        o_1 = self.act(o_1)
        o_2 = torch.cat([o_1, x2], 1)
        o_2 = self.act(self.denorm1d_2(self.deconv2(o_2)))
        o_3 = torch.cat([o_2, x1], 1)
        o_3 = self.deconv3(o_3)
        out = o_3.reshape(b,t,c,-1)
        mask = out.permute(0,3,1,2)
        crm_out_real = x_ref[..., 0]*mask[..., 0] - x_ref[..., 1]*mask[..., 1]
        crm_out_imag = x_ref[..., 0]*mask[..., 1] + x_ref[..., 1]*mask[..., 0]
        crm_out = torch.stack([crm_out_real, crm_out_imag], -1)

        return crm_out
    
class FSPEN(DPCRN_Light):
    def __init__(self, in_ch=2, out_ch=2, filter_size=[4,16,32], kernel_size=[6,8,6],stride_size=[2,2,2], numUnits=16) -> None:
        super().__init__(in_ch, out_ch, filter_size, kernel_size, stride_size, numUnits)
        self.subbandEncode = SUBBAND_ENCODE()
        self.subbandDecode = SUBBAND_DECODE()
        self.featureMerge = nn.Sequential(nn.Linear(filter_size[-1]*2+2, filter_size[-1]), nn.ELU())
        self.featureSplit = nn.Sequential(nn.Linear(filter_size[-1], filter_size[-1]*2+2), nn.ELU())
        

    def forward(self, x):
        # bftc 
        x_ref = x
        x_mag = torch.view_as_complex(x.contiguous()).abs().unsqueeze(-1) # bft1
        x_mag_unproc = x_mag.permute(0,2,3,1) # btcf
        
        x = x.permute(0, 2, 3, 1)
        b, t, c, f = x.shape
        x1 = x.reshape(b*t, c, f) # b*t c f 
        x_mag_unproc = x_mag_unproc.reshape(b*t, -1, f)

        # subband encode
        subband_encode = self.subbandEncode(x_mag_unproc)
        subband_feature = torch.cat(subband_encode, -1)
        

        # fullband encode
        # x1 = x1.permute(0,2,1) 
        x1 = self.conv1d_1(x1)
        x1 = self.act(self.norm1d_1(x1))
        x2 = self.conv1d_2(x1)
        x2 = self.act(self.norm1d_2(x2))
        x3 = self.conv1d_3(x2)
        x3 = self.act(self.norm1d_3(x3))
        x4 = self.conv1X1d(x3)

        # full subband feature merge
        fullsubfeat = torch.cat([x4, subband_feature], -1)
        feat_merge = self.featureMerge(self.conv1_fm(fullsubfeat))

        # x4 = self.conv1_fm(x4)

        x_rnn_in = feat_merge.reshape(b,t,self.filter_size[-1]//2, -1)
        x_rnn_in = x_rnn_in.permute(0, 2, 1, 3) # b c t f
        rnn_out = self.dprnn(x_rnn_in)
        rnn_out = self.dprnn1(rnn_out)
        rnn_out = self.dprnn2(rnn_out)

        rnn_out = rnn_out.permute(0, 2, 1, 3)
        rnn_out = rnn_out.reshape(b*t, self.filter_size[-1]//2, -1) # b*t c f
        rnn_out = self.deconv1_fm(rnn_out)

        feat_split = self.featureSplit(rnn_out)
        fullband_split = feat_split[...,:self.filter_size[-1]]
        subband_split = feat_split[...,self.filter_size[-1]:]

        subband_decode = self.subbandDecode(subband_encode, subband_split)
        mag_gain = subband_decode.reshape(b,t,-1).permute(0,2, 1).unsqueeze(-1)
        o_1 = torch.cat([fullband_split, x3], 1)
        o_1 = self.deconv1(o_1)
        o_1 = self.denorm1d_1(o_1)
        o_1 = self.act(o_1)
        o_2 = torch.cat([o_1, x2], 1)
        o_2 = self.act(self.denorm1d_2(self.deconv2(o_2)))
        o_3 = torch.cat([o_2, x1], 1)
        o_3 = self.deconv3(o_3)
        out = o_3.reshape(b,t,c,-1)
        mask = out.permute(0,3,1,2)
        crm_out_real = x_ref[..., 0]*mask[..., 0] - x_ref[..., 1]*mask[..., 1]
        crm_out_imag = x_ref[..., 0]*mask[..., 1] + x_ref[..., 1]*mask[..., 0]
        # mag_enh = x_mag * mag_gain
        
        crm_abs = torch.view_as_complex(mask.contiguous()).abs().unsqueeze(-1)
        crm_enh = (mag_gain + crm_abs)*0.5/crm_abs

        crm_out = torch.stack([crm_out_real, crm_out_imag], -1) * crm_enh


        return crm_out





def test_DPCRN_light():
    x = torch.randn(2,257,3,2)
    model = DPCRN_Light()
    out = model(x)
    print('sc')

def test_FSPEN():
    x = torch.randn(2,257,3,2)
    model = FSPEN()
    out = model(x)
    print('sc')


if __name__ == "__main__":
    test_FSPEN()