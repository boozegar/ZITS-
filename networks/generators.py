import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.ffc import ResnetBlock_remove_IN, GFBlock
from networks.layers import GateConv, MaskedSinusoidalPositionalEmbedding, MultiLabelEmbedding, ResnetBlock
from networks.van import VANBlock


class SpatialZeroRAAdapter(nn.Module):
    """Predicts a zero-initialized spatial correction for ZeroRA injection."""

    def __init__(
        self,
        channels,
        hidden_ratio=0.25,
        max_hidden_channels=64,
        delta_scale=1.0,
        output_channels=1,
    ):
        super().__init__()
        hidden_channels = max(8, min(max_hidden_channels, int(channels * hidden_ratio)))
        self.delta_scale = float(delta_scale)
        self.project_in = nn.Conv2d(channels + 1, hidden_channels, kernel_size=1)
        self.spatial = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
        )
        self.act = nn.SiLU()
        self.project_out = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)
        nn.init.zeros_(self.project_out.weight)
        nn.init.zeros_(self.project_out.bias)

    def forward(self, feature, mask):
        if mask.shape[-2:] != feature.shape[-2:]:
            mask = F.adaptive_max_pool2d(mask, feature.shape[-2:])
        mask = mask.to(dtype=feature.dtype)
        delta = self.project_in(torch.cat([feature, mask], dim=1))
        delta = self.act(delta)
        delta = self.act(self.spatial(delta))
        delta = torch.tanh(self.project_out(delta)) * self.delta_scale
        return delta * mask


class DenseGuidanceDecoderAdapter(nn.Module):
    """Maps raw dense structure guidance into a zero-initialized decoder residual."""

    def __init__(
        self,
        guidance_channels,
        output_channels,
        hidden_channels=32,
        feature_channels=None,
    ):
        super().__init__()
        self.project_in = nn.Conv2d(
            guidance_channels + 1,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.spatial = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
        )
        self.feature_project = (
            nn.Conv2d(feature_channels, hidden_channels, kernel_size=1)
            if feature_channels is not None
            else None
        )
        if self.feature_project is not None:
            nn.init.zeros_(self.feature_project.weight)
            nn.init.zeros_(self.feature_project.bias)
        self.act = nn.SiLU()
        self.project_out = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)
        nn.init.zeros_(self.project_out.weight)
        nn.init.zeros_(self.project_out.bias)

    def forward(self, guidance, mask, output_size, feature=None):
        guidance = F.interpolate(
            guidance,
            size=output_size,
            mode='bilinear',
            align_corners=False,
        )
        mask = F.adaptive_max_pool2d(mask, output_size).to(dtype=guidance.dtype)
        delta = self.project_in(torch.cat([guidance, mask], dim=1))
        if self.feature_project is not None:
            if feature is None or feature.shape[-2:] != output_size:
                raise ValueError('Feature-conditioned adapter requires matching decoder features')
            delta = delta + self.feature_project(feature)
        delta = self.act(delta)
        delta = self.act(self.spatial(delta))
        return self.project_out(delta) * mask


def _configure_dense_decoder_adapters(module, config):
    dense_config = config.get('dense_decoder_zero_ra', {})
    module.use_dense_decoder_zero_ra = bool(dense_config.get('enabled', False))
    if not module.use_dense_decoder_zero_ra:
        return
    guidance_channels = int(dense_config.get('guidance_channels', 4))
    hidden_channels = int(dense_config.get('hidden_channels', 32))
    feature_conditioned = bool(dense_config.get('feature_conditioned', False))
    guidance_names = tuple(dense_config.get(
        'guidance_names',
        ('edge', 'gradientx', 'gradienty', 'line'),
    ))
    enabled_stages = tuple(int(index) for index in dense_config.get(
        'enabled_stages',
        (0, 1, 2),
    ))
    if guidance_channels <= 0 or hidden_channels <= 0:
        raise ValueError('Dense decoder adapter channels must be positive')
    allowed_guidance = {'edge', 'gradientx', 'gradienty', 'line'}
    if len(guidance_names) != len(set(guidance_names)) or not set(guidance_names) <= allowed_guidance:
        raise ValueError('Invalid dense decoder guidance_names')
    if len(enabled_stages) != len(set(enabled_stages)) or not set(enabled_stages) <= {0, 1, 2}:
        raise ValueError('Invalid dense decoder enabled_stages')
    module.dense_decoder_guidance_names = guidance_names
    module.dense_decoder_enabled_stages = enabled_stages
    module.dense_decoder_feature_conditioned = feature_conditioned
    module.dense_guidance_adapters = nn.ModuleList([
        DenseGuidanceDecoderAdapter(
            guidance_channels,
            channels,
            hidden_channels,
            feature_channels=channels if feature_conditioned else None,
        )
        for channels in (256, 128, 64)
    ])


def _apply_dense_decoder_adapter(module, index, feature, guidance, mask):
    if (
        not module.use_dense_decoder_zero_ra
        or index not in module.dense_decoder_enabled_stages
    ):
        return feature
    if guidance is None or mask is None:
        raise ValueError('Dense guidance and mask are required for decoder adapters')
    return feature + module.dense_guidance_adapters[index](
        guidance,
        mask,
        feature.shape[-2:],
        feature=feature,
    )


def _select_dense_decoder_guidance(module, batch):
    guidance_names = module.dense_decoder_guidance_names
    return torch.cat([
        batch[name] if name in guidance_names else torch.zeros_like(batch[name])
        for name in ('edge', 'gradientx', 'gradienty', 'line')
    ], dim=1)


class myFFCResNetGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config['activation'] == 'relu':
            print('use relu')
            act = nn.ReLU
        elif config['activation'] == 'swish':
            print('use swish')
            act = nn.SiLU
        else:
            print('use identity')
            act = nn.Identity

        self.enconv1 = nn.Sequential(*[nn.ReflectionPad2d(3),
                                       nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
                                       nn.BatchNorm2d(64), act()])
        self.enconv2 = nn.Sequential(*[nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(128), act()])
        self.enconv3 = nn.Sequential(*[nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(256), act()])
        self.enconv4 = nn.Sequential(*[nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(512), act()])

        blocks = []
        # resnet blocks
        if config['use_GFBlock'] is False:
            print('use FFC')
            for i in range(9):
                blocks.append(ResnetBlock_remove_IN(512, 1, activation_layer=act))
        else:
            print('use Global Filter')
            for i in range(int(config['num_GFBlock'])):
                blocks.append(GFBlock(dim=512))

        self.middle = nn.Sequential(*blocks)

        self.deconv1 = nn.Sequential(*[nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                                       nn.BatchNorm2d(256), act()])
        self.deconv2 = nn.Sequential(*[nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                                       nn.BatchNorm2d(128), act()])
        self.deconv3 = nn.Sequential(*[nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                                       nn.BatchNorm2d(64), act()])
        self.deconv4 = nn.Sequential(*[nn.ReflectionPad2d(3), nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0)])

        self.act_last = nn.Tanh()
        _configure_dense_decoder_adapters(self, config)

    def forward(self, x):
        x = self.enconv1(x)
        x = self.enconv2(x)
        x = self.enconv3(x)
        x = self.enconv4(x)

        x = self.middle(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.act_last(x)
        return x

    def forward_with_prior(
        self,
        x,
        rel_pos_emb,
        direct_emb,
        str_feats,
        dense_guidance=None,
        mask=None,
    ):
        x = self.enconv1[:2](x)
        inp = x.to(torch.float32) + rel_pos_emb + direct_emb
        x = self.enconv1[2:](inp)

        x = self.enconv2[:1](x + str_feats[0])
        x = self.enconv2[1:](x)

        x = self.enconv3[:1](x + str_feats[1])
        x = self.enconv3[1:](x)

        x = self.enconv4[:1](x + str_feats[2])
        x = self.enconv4[1:](x)

        x = self.middle(x + str_feats[3])

        x = self.deconv1(x)
        x = _apply_dense_decoder_adapter(self, 0, x, dense_guidance, mask)
        x = self.deconv2(x)
        x = _apply_dense_decoder_adapter(self, 1, x, dense_guidance, mask)
        x = self.deconv3(x)
        x = _apply_dense_decoder_adapter(self, 2, x, dense_guidance, mask)
        x = self.deconv4(x)
        x = self.act_last(x)

        return x


class VANFFCResNetGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        print('VANFFCResNetGenerator initialized')

        if config['activation'] == 'relu':
            print('use relu')
            act = nn.ReLU
        elif config['activation'] == 'swish':
            print('use swish')
            act = nn.SiLU
        else:
            print('use identity')
            act = nn.Identity

        self.enconv1 = nn.Sequential(*[nn.ReflectionPad2d(3),
                                       nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
                                       nn.BatchNorm2d(64), act(True)])
        self.enconv2 = nn.Sequential(*[nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(128), act(True),
                                       VANBlock(dim=128, kernel_size=config['van_kernel_size'],
                                                dilation=config['van_dilation'], act_layer=act)])
        self.enconv3 = nn.Sequential(*[nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(256), act(True),
                                       VANBlock(dim=256, kernel_size=config['van_kernel_size'],
                                                dilation=config['van_dilation'], act_layer=act)])
        self.enconv4 = nn.Sequential(*[nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(512), act(True),
                                       VANBlock(dim=512, kernel_size=config['van_kernel_size'],
                                                dilation=config['van_dilation'], act_layer=act)])

        blocks = []
        # resnet blocks
        if config['use_GFBlock'] is False:
            if config['use_VAN_between_FFC'] is False:
                print('use pure FFC')
                for i in range(9):
                    blocks.append(ResnetBlock_remove_IN(512, 1, activation_layer=act))

        self.middle = nn.Sequential(*blocks)

        self.deconv1 = nn.Sequential(*[VANBlock(dim=512, kernel_size=config['van_kernel_size'],
                                                dilation=config['van_dilation'], act_layer=act),
                                       nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                                       nn.BatchNorm2d(256), act(True)])
        self.deconv2 = nn.Sequential(*[VANBlock(dim=256, kernel_size=config['van_kernel_size'],
                                                dilation=config['van_dilation'], act_layer=act),
                                       nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                                       nn.BatchNorm2d(128), act(True)])
        self.deconv3 = nn.Sequential(*[VANBlock(dim=128, kernel_size=config['van_kernel_size'],
                                                dilation=config['van_dilation'], act_layer=act),
                                       nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                                       nn.BatchNorm2d(64), act(True)])
        self.deconv4 = nn.Sequential(*[nn.ReflectionPad2d(3), nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0)])

        self.act_last = nn.Tanh()
        _configure_dense_decoder_adapters(self, config)

    def forward(self, x):
        x = self.enconv1(x)
        x = self.enconv2(x)
        x = self.enconv3(x)
        x = self.enconv4(x)

        x = self.middle(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.act_last(x)
        return x

    def forward_with_prior(
        self,
        x,
        rel_pos_emb,
        direct_emb,
        str_feats,
        dense_guidance=None,
        mask=None,
    ):
        x = self.enconv1[:2](x)
        inp = x.to(torch.float32) + rel_pos_emb + direct_emb
        x = self.enconv1[2:](inp)

        x = self.enconv2[:1](x + str_feats[0])
        x = self.enconv2[1:](x)

        x = self.enconv3[:1](x + str_feats[1])
        x = self.enconv3[1:](x)

        x = self.enconv4[:1](x + str_feats[2])
        x = self.enconv4[1:](x)

        x = self.middle(x + str_feats[3])

        x = self.deconv1(x)
        x = _apply_dense_decoder_adapter(self, 0, x, dense_guidance, mask)
        x = self.deconv2(x)
        x = _apply_dense_decoder_adapter(self, 1, x, dense_guidance, mask)
        x = self.deconv3(x)
        x = _apply_dense_decoder_adapter(self, 2, x, dense_guidance, mask)
        x = self.deconv4(x)
        x = self.act_last(x)

        return x


class StructureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = GateConv(in_channels=config['prior_ch'], out_channels=64, kernel_size=7, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.ReLU(True)

        self.conv2 = GateConv(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        blocks = []
        # resnet blocks
        for i in range(3):
            blocks.append(ResnetBlock(input_dim=512, out_dim=None, dilation=2))

        self.middle = nn.Sequential(*blocks)
        self.alpha1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        self.convt1 = GateConv(512, 256, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt1 = nn.BatchNorm2d(256)
        self.alpha2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        self.convt2 = GateConv(256, 128, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt2 = nn.BatchNorm2d(128)
        self.alpha3 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        self.convt3 = GateConv(128, 64, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt3 = nn.BatchNorm2d(64)
        self.alpha4 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        spatial_config = self.config.get('spatial_zero_ra', {})
        self.use_spatial_zero_ra = bool(spatial_config.get('enabled', False))
        if self.use_spatial_zero_ra:
            self.spatial_zero_ra_mode = spatial_config.get('mode', 'scalar_gate')
            if self.spatial_zero_ra_mode not in ('scalar_gate', 'channel_residual'):
                raise ValueError(f"Unsupported spatial_zero_ra mode: {self.spatial_zero_ra_mode}")
            adapter_args = {
                'hidden_ratio': spatial_config.get('hidden_ratio', 0.25),
                'max_hidden_channels': spatial_config.get('max_hidden_channels', 64),
                'delta_scale': spatial_config.get('delta_scale', 1.0),
            }
            output_channels = (
                [1, 1, 1, 1]
                if self.spatial_zero_ra_mode == 'scalar_gate'
                else [512, 256, 128, 64]
            )
            self.spatial_adapters = nn.ModuleList([
                SpatialZeroRAAdapter(512, output_channels=output_channels[0], **adapter_args),
                SpatialZeroRAAdapter(256, output_channels=output_channels[1], **adapter_args),
                SpatialZeroRAAdapter(128, output_channels=output_channels[2], **adapter_args),
                SpatialZeroRAAdapter(64, output_channels=output_channels[3], **adapter_args),
            ])

        if self.config['rezero_for_mpe']:
            self.rel_pos_emb = MaskedSinusoidalPositionalEmbedding(num_embeddings=self.config['rel_pos_num'],
                                                                   embedding_dim=64)
            self.direct_emb = MultiLabelEmbedding(num_positions=4, embedding_dim=64)
            self.alpha5 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
            self.alpha6 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

    def _apply_zero_ra(self, feature, alpha, mask, scale_index):
        if not self.use_spatial_zero_ra:
            return feature * alpha
        spatial_delta = self.spatial_adapters[scale_index](feature, mask)
        if self.spatial_zero_ra_mode == 'channel_residual':
            return feature * alpha + spatial_delta
        return feature * (alpha + spatial_delta)

    def forward(self, x, rel_pos=None, direct=None, mask=None):
        if self.use_spatial_zero_ra and mask is None:
            raise ValueError('mask is required when spatial_zero_ra is enabled')
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x.to(torch.float32))
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x.to(torch.float32))
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x.to(torch.float32))
        x = self.act(x)

        x = self.conv4(x)
        x = self.bn4(x.to(torch.float32))
        x = self.act(x)

        return_feats = []
        x = self.middle(x)
        return_feats.append(self._apply_zero_ra(x, self.alpha1, mask, 0))

        x = self.convt1(x)
        x = self.bnt1(x.to(torch.float32))
        x = self.act(x)
        return_feats.append(self._apply_zero_ra(x, self.alpha2, mask, 1))

        x = self.convt2(x)
        x = self.bnt2(x.to(torch.float32))
        x = self.act(x)
        return_feats.append(self._apply_zero_ra(x, self.alpha3, mask, 2))

        x = self.convt3(x)
        x = self.bnt3(x.to(torch.float32))
        x = self.act(x)
        return_feats.append(self._apply_zero_ra(x, self.alpha4, mask, 3))

        return_feats = return_feats[::-1]

        b, h, w = rel_pos.shape
        rel_pos = rel_pos.reshape(b, h * w)
        rel_pos_emb = self.rel_pos_emb(rel_pos).reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.alpha5
        direct = direct.reshape(b, h * w, 4).to(torch.float32)
        direct_emb = self.direct_emb(direct).reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.alpha6

        return return_feats, rel_pos_emb, direct_emb


class FTRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if 'not_use_van' in self.config and self.config['not_use_van'] is True:
            self.G = myFFCResNetGenerator(config)
        else:
            self.G = VANFFCResNetGenerator(config)
        self.GCs = StructureEncoder(config)

    def forward(self, batch):
        img = batch['image']
        mask = batch['mask']
        masked_img = img * (1 - mask)

        masked_img = torch.cat([masked_img, mask], dim=1)
        if self.config['use_gradient'] is True:
            if 'not_use_line' in self.config and self.config['not_use_line'] is True:
                masked_str = torch.cat([batch['edge'], batch['gradientx'], batch['gradienty'], mask], dim=1)
                dense_guidance = torch.cat(
                    [batch['edge'], batch['gradientx'], batch['gradienty']],
                    dim=1,
                )
            else:
                masked_str = torch.cat([batch['edge'], batch['gradientx'], batch['gradienty'], batch['line'], mask], dim=1)
                dense_guidance = torch.cat(
                    [batch['edge'], batch['gradientx'], batch['gradienty'], batch['line']],
                    dim=1,
                )
        else:
            masked_str = torch.cat([batch['edge'], batch['line'], mask], dim=1)
            dense_guidance = torch.cat([batch['edge'], batch['line']], dim=1)

        if self.G.use_dense_decoder_zero_ra:
            dense_guidance = _select_dense_decoder_guidance(self.G, batch)

        str_feats, rel_pos_emb, direct_emb = self.GCs(
            masked_str,
            batch['rel_pos'],
            batch['direct'],
            mask=mask,
        )
        gen_img = self.G.forward_with_prior(
            masked_img.to(torch.float32),
            rel_pos_emb,
            direct_emb,
            str_feats,
            dense_guidance=dense_guidance,
            mask=mask,
        )

        return gen_img
