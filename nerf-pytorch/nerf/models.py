import torch
import torch.nn.functional as F


class VeryTinyNeRFModel(torch.nn.Module):
    r"""Define a "very tiny" NeRF model comprising three fully connected layers.
    """

    def __init__(self, filter_size=128, num_encoding_functions=6, use_viewdirs=True):
        super(VeryTinyNeRFModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        if use_viewdirs is True:
            self.viewdir_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.viewdir_encoding_dims = 0
        # Input layer (default: 65 -> 128)
        self.layer1 = torch.nn.Linear(
            self.xyz_encoding_dims + self.viewdir_encoding_dims, filter_size
        )
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        # Layer 3 (default: 128 -> 4)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class MultiHeadNeRFModel(torch.nn.Module):
    r"""Define a "multi-head" NeRF model (radiance and RGB colors are predicted by
    separate heads).
    """

    def __init__(self, hidden_size=128, num_encoding_functions=6, use_viewdirs=True):
        super(MultiHeadNeRFModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        if use_viewdirs is True:
            self.viewdir_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.viewdir_encoding_dims = 0
        # Input layer (default: 39 -> 128)
        self.layer1 = torch.nn.Linear(self.xyz_encoding_dims, hidden_size)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        # Layer 3_1 (default: 128 -> 1): Predicts radiance ("sigma")
        self.layer3_1 = torch.nn.Linear(hidden_size, 1)
        # Layer 3_2 (default: 128 -> 1): Predicts a feature vector (used for color)
        self.layer3_2 = torch.nn.Linear(hidden_size, hidden_size)

        # Layer 4 (default: 39 + 128 -> 128)
        self.layer4 = torch.nn.Linear(
            self.viewdir_encoding_dims + hidden_size, hidden_size
        )
        # Layer 5 (default: 128 -> 128)
        self.layer5 = torch.nn.Linear(hidden_size, hidden_size)
        # Layer 6 (default: 128 -> 3): Predicts RGB color
        self.layer6 = torch.nn.Linear(hidden_size, 3)

        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x, view = x[..., : self.xyz_encoding_dims], x[..., self.xyz_encoding_dims :]
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        sigma = self.layer3_1(x)
        feat = self.relu(self.layer3_2(x))
        x = torch.cat((feat, view), dim=-1)
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.layer6(x)
        return torch.cat((x, sigma), dim=-1)


class ReplicateNeRFModel(torch.nn.Module):
    r"""NeRF model that follows the figure (from the supp. material of NeRF) to
    every last detail. (ofc, with some flexibility)
    """

    def __init__(
        self,
        hidden_size=256,
        num_layers=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
    ):
        super(ReplicateNeRFModel, self).__init__()
        # xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions

        self.dim_xyz = (3 if include_input_xyz else 0) + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = (3 if include_input_dir else 0) + 2 * 3 * num_encoding_fn_dir

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_alpha = torch.nn.Linear(hidden_size, 1)

        self.layer4 = torch.nn.Linear(hidden_size + self.dim_dir, hidden_size // 2)
        self.layer5 = torch.nn.Linear(hidden_size // 2, hidden_size // 2)
        self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        xyz, direction = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        x_ = self.relu(self.layer1(xyz))
        x_ = self.relu(self.layer2(x_))
        feat = self.layer3(x_)
        alpha = self.fc_alpha(x_)
        y_ = self.relu(self.layer4(torch.cat((feat, direction), dim=-1)))
        y_ = self.relu(self.layer5(y_))
        rgb = self.fc_rgb(y_)
        return torch.cat((rgb, alpha), dim=-1)


class PaperNeRFModel(torch.nn.Module):
    r"""Implements the NeRF model as described in Fig. 7 (appendix) of the
    arXiv submission (v0). """

    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
    ):
        super(PaperNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir

        self.layers_xyz = torch.nn.ModuleList()
        self.use_viewdirs = use_viewdirs
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz, 256))
        for i in range(1, 8):
            if i == 4:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + 256, 256))
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 128))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        for i in range(8):
            if i == 4:
                x = self.layers_xyz[i](torch.cat((xyz, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        if self.use_viewdirs:
            x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        else:
            x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)


class FlexibleNeRFModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=4,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        color_channel=3
    ):
        super(FlexibleNeRFModel, self).__init__()

        self.num_layers = num_layers

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        #print(self.dim_xyz, self.dim_dir)
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                #print(self.dim_xyz + hidden_size)
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)

            self.fc_rgb = torch.nn.Linear(hidden_size // 2, color_channel)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, color_channel+1)

        self.relu = torch.nn.functional.relu

    def forward(self, xin):
        if self.use_viewdirs:
            xyz, view = xin[..., : self.dim_xyz], xin[..., self.dim_xyz :]
        else:
            xyz = xin[..., : self.dim_xyz]
        
        x = self.layer1(xyz)
        
        for i in range(len(self.layers_xyz)):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != self.num_layers - 1
            ):
                #print("enter")
                x = torch.cat((x, xyz), dim=-1)
            #print(x.shape, len(self.layers_xyz), self.layers_xyz[i], i, self.num_layers - 1)
            x = self.relu(self.layers_xyz[i](x))
        
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)

            return torch.cat((rgb, alpha), dim=-1)
            
        else:
            return self.fc_out(x)

    def forwardf(self, xin):

        if self.use_viewdirs:
            xyz, view = xin[..., : self.dim_xyz], xin[..., self.dim_xyz :]
        else:
            xyz = xin[..., : self.dim_xyz]

        x1 = self.layer1(xyz)
        print(x1.requires_grad)
        print(x1.shape)
        x_t = x1.clone()
        
        for i in range(len(self.layers_xyz)):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != self.num_layers - 1
            ):
                #print("enter")
                x_t = torch.cat((x_t, xyz), dim=-1)
            #print(x.shape, len(self.layers_xyz), self.layers_xyz[i], i, self.num_layers - 1)
            x_t = self.relu(self.layers_xyz[i](x_t))
        x_test = x_t
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x_test))
            alpha = self.fc_alpha(x_test)
            x_v = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x_v = self.relu(l(x_v))
            rgb = self.fc_rgb(x_v)

            d_output = torch.ones_like(alpha, requires_grad=False, device=alpha.device)
            gradients = torch.autograd.grad(
                                            outputs=alpha,
                                            inputs=xin,
                                            grad_outputs=d_output,
                                            create_graph=True,
                                            retain_graph=True,
                                            only_inputs=True,
                                            #allow_unused=True
                                            )[0]
            print(gradients.shape)
            assert 1==0

            return torch.cat((rgb, alpha), dim=-1)
            
        else:
            return self.fc_out(x)

class FlexibleIRNeRFModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=4,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        use_alpha=False,
        color_channel=3
    ):
        super(FlexibleIRNeRFModel, self).__init__()

        self.num_layers = num_layers
        self.use_alpha = use_alpha

        include_input_xyz = 3 * 2 if include_input_xyz else 0
        include_input_dir = 3 * 2 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz * 2
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir * 2


        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        self.attenuation = torch.nn.parameter.Parameter(torch.tensor([10.]),
            requires_grad=True)
        
        self.layer1 = torch.nn.Linear(1+self.dim_xyz//2, hidden_size)
        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                #print(self.dim_xyz + hidden_size)
                self.layers_xyz.append(
                    torch.nn.Linear(1+self.dim_xyz//2 + hidden_size, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            #print(self.dim_dir//2,hidden_size)
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir//2 + hidden_size, hidden_size // 2)
            )
            if use_alpha:
                self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, color_channel)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, color_channel+1)

        self.relu = torch.nn.functional.relu

        #self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layers_ir = torch.nn.ModuleList()
        self.layers_ir1 = torch.nn.Linear(self.dim_dir // 2, hidden_size)
        for i in range(num_layers//2 - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                #print(self.dim_xyz + hidden_size)
                self.layers_ir.append(
                    torch.nn.Linear(self.dim_dir // 2 + hidden_size, hidden_size)
                )
            else:
                self.layers_ir.append(torch.nn.Linear(hidden_size, hidden_size))
        
        self.layers_ir_out = torch.nn.Linear(hidden_size, color_channel)
        


    def forward(self, x):
        #print(x.shape)
        # x[0,:3], x[0,63:66], x[0,126:129], x[0,153:156]
        #assert 1==0
        # incident intensity
        #print(self.layers_ir)
        #dir_ir = x[..., 153:156]
        dir_ir = x[..., 153:]
        xyz_ir = x[..., 63:66]
        xyz, view = x[...,:63], x[...,126:153]
        #print(xyz_ir.shape)
        #assert 1==0
        x = self.layers_ir1(dir_ir)
        for i in range(len(self.layers_ir)):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != self.num_layers - 1
            ):
                x = torch.cat((x, dir_ir), dim=-1)
            x = self.relu(self.layers_ir[i](x))
        
        ir_intensity = self.layers_ir_out(x)
        
        dist = torch.norm(xyz_ir, dim=-1)
        attenuation_eff = torch.clip(self.attenuation/dist, 0.,1.)[...,None]
        ir_in_intensity = attenuation_eff*ir_intensity
        #print(ir_intensity.shape, attenuation_eff.shape, ir_in_intensity.shape)
        #assert 1==0
        
        """
        if self.use_viewdirs:
            #print('use')
            #x[..., : self.dim_xyz], x[..., self.dim_xyz :]
            #print(view.shape)
            #print(xyz[0,:3],xyz[0,63:66], view[0,:3])
            #assert 1==0
            xyz, view = x[...,:63], x[...,126:153]
        else:
            xyz = x[..., : self.dim_xyz]
        """
        xyz = torch.cat((xyz, ir_in_intensity), dim=-1)
        x = self.layer1(xyz)
        for i in range(len(self.layers_xyz)):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != self.num_layers - 1
            ):
                #print("enter")
                x = torch.cat((x, xyz), dim=-1)
            #print(x.shape, len(self.layers_xyz), self.layers_xyz[i], i, self.num_layers - 1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            if self.use_alpha:
                alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                #print(feat.shape, view.shape, x.shape)
                #assert 1==0
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            if self.use_alpha:
                return torch.cat((rgb, alpha), dim=-1)
            else:
                return rgb

        else:
            return self.fc_out(x)

class RadianceFuseModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=4,
        hidden_size=128
    ):
        super(RadianceFuseModel, self).__init__()
        self.num_layers = num_layers

        self.layer1 = torch.nn.Linear(2, hidden_size)
        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))
        self.fc_out = torch.nn.Linear(hidden_size, 1)

        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.layer1(x)
        for i in range(len(self.layers_xyz)):
            x = self.relu(self.layers_xyz[i](x))
            
        return self.fc_out(x)

class FlexibleIRReflectanceModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=4,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        include_input_xyz=True,
        #use_viewdirs=True,
        #use_alpha=False,
        color_channel=3,
        H=1080,
        W=1920,
        ir_intrinsic=None,
        ir_extrinsic=None,
        ir_gt=None
    ):
        super(FlexibleIRReflectanceModel, self).__init__()
        self.static_ir_pat = False

        self.num_layers = num_layers
        #self.use_alpha = use_alpha

        include_input_xyz = 3  if include_input_xyz else 0
        #include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        #self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.skip_connect_every = skip_connect_every
        #if not use_viewdirs:
        #    self.dim_dir = 0

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                #print(self.dim_xyz + hidden_size)
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))


        self.fc_out = torch.nn.Linear(hidden_size, 5)

        #ir_pattern_ts = torch.zeros([H,W], requires_grad=False)
        #ir_pattern_ts[:int(H/2),:] = 1.
        self.H = 2*H
        self.W = 2*W
        self.ir_pattern = torch.nn.parameter.Parameter(torch.zeros([2*W,2*H]), requires_grad=True)
        if ir_gt is not None:
            self.ir_pattern = torch.load(ir_gt)
            self.ir_pattern.requires_grad = False
        self.ir_intrinsic = ir_intrinsic
        #self.light_attenuation_coeff = torch.nn.parameter.Parameter(torch.zeros([1]))

        #self.ir_extrinsic = ir_extrinsic
        #self.ir_pattern = torch.nn.parameter.Parameter(ir_pattern_ts, requires_grad=False)
        

        self.relu = torch.nn.functional.relu

        self.act_normal = torch.nn.Tanh()
        self.act_brdf = torch.nn.Sigmoid()
        self.act_coef = torch.nn.Softplus()

    def get_light(self, surface_xyz, light_extrinsic):
        w2ir = torch.linalg.inv(light_extrinsic)
        surface_xyz = surface_xyz.transpose(0,1)
        #print((torch.matmul(w2ir[:3,:3], surface_xyz)+w2ir[:3,3][...,None]).shape)
        #assert 1==0
        ir_xyz = torch.matmul(w2ir[:3,:3], surface_xyz) + w2ir[:3,3][...,None] # 3xn

        #distance = torch.norm(ir_xyz, dim=0) # n
        #attenuation_multiplier = 1./(1. + self.act_coef(self.light_attenuation_coeff)*(distance**2))
        #print(distance[0], ir_xyz[:,0])
        #assert 1==0
        #irl2s = (surface_xyz - self.ir_extrinsic[:3,3]).detach()
        irl_pix_homo = torch.matmul(self.ir_intrinsic, ir_xyz)
        irl_pix = (irl_pix_homo / irl_pix_homo[-1,:])[:2,:]
        #irl_pix = torch.zeros([2,1]).cuda()
        #irl_pix[:,0] = torch.tensor([0, 0])
        irl_pix = irl_pix.transpose(0,1)[None,None,...]
        irl_pix[:,:,:,0] = ((irl_pix[:,:,:,0]/(self.W-1)) - 0.5) * 2.
        irl_pix[:,:,:,1] = ((irl_pix[:,:,:,1]/(self.H-1)) - 0.5) * 2.

        grid = torch.zeros(irl_pix.shape).cuda()
        grid[:,:,:,0] = irl_pix[:,:,:,1]
        grid[:,:,:,1] = irl_pix[:,:,:,0]
        
        #print(torch.max(grid[:,:,:,0]), torch.min(grid[:,:,:,0]), torch.max(grid[:,:,:,1]), torch.min(grid[:,:,:,1]))
        #assert 1==0
        #print(grid)
        ir_pattern = torch.nn.functional.softplus(self.ir_pattern, beta=5)
        ir_pattern = ir_pattern[None,None,...]
        #ir_pattern = torch.rand(ir_pattern.shape).cuda()

        #test_grid = torch.zeros([1,1,1,2]).cuda()
        #test_grid[0,0,0,:] = torch.tensor([-1,-1])
        if self.static_ir_pat:
            max_ir = torch.max(ir_pattern)
            min_ir = torch.min(ir_pattern)
            
            mid_ir = 0.5*(max_ir + min_ir)
            light_out = F.grid_sample(ir_pattern, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
  
            light_out[light_out >= mid_ir] = max_ir
            
            light_out[light_out < mid_ir] = min_ir
    
            light_out = light_out[0,0,0,:]

        else:
            light_out = F.grid_sample(ir_pattern, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
            #print(light_out.shape, attenuation_multiplier.shape)
            #assert 1==0
            light_out = light_out[0,0,0,:]#*attenuation_multiplier
        
        #print(light_out.shape)
        #assert 1==0
        surf2l_ray = (light_extrinsic[:3,3][...,None] - surface_xyz).transpose(0,1)

        

        surf2l = F.normalize(surf2l_ray,p=2.0,dim=1)


        #surf2l = -l2surf
        return light_out, surf2l

    """
    def get_light(self, surface_xyz, light_extrinsic):
        w2ir = torch.linalg.inv(light_extrinsic)
        surface_xyz = surface_xyz.transpose(0,1)
        #print((torch.matmul(w2ir[:3,:3], surface_xyz)+w2ir[:3,3][...,None]).shape)
        #assert 1==0
        ir_xyz = torch.matmul(w2ir[:3,:3], surface_xyz) + w2ir[:3,3][...,None] # 3xn

        distance = torch.norm(ir_xyz, dim=0) # n
        attenuation_multiplier = 1./(1. + self.act_coef(self.light_attenuation_coeff)*(distance**2))
        #print(distance[0], ir_xyz[:,0])
        #assert 1==0
        #irl2s = (surface_xyz - self.ir_extrinsic[:3,3]).detach()
        irl_pix_homo = torch.matmul(self.ir_intrinsic, ir_xyz)
        irl_pix = (irl_pix_homo / irl_pix_homo[-1,:])[:2,:]
        #irl_pix = torch.zeros([2,1]).cuda()
        #irl_pix[:,0] = torch.tensor([0, 0])
        irl_pix = irl_pix.transpose(0,1)[None,None,...]
        irl_pix[:,:,:,0] = ((irl_pix[:,:,:,0]/(self.W-1)) - 0.5) * 2.
        irl_pix[:,:,:,1] = ((irl_pix[:,:,:,1]/(self.H-1)) - 0.5) * 2.

        grid = torch.zeros(irl_pix.shape).cuda()
        grid[:,:,:,0] = irl_pix[:,:,:,1]
        grid[:,:,:,1] = irl_pix[:,:,:,0]
        
        #print(torch.max(grid[:,:,:,0]), torch.min(grid[:,:,:,0]), torch.max(grid[:,:,:,1]), torch.min(grid[:,:,:,1]))
        #assert 1==0
        #print(grid)
        ir_pattern = torch.nn.functional.softplus(self.ir_pattern, beta=5)
        ir_pattern = ir_pattern[None,None,...]
        #ir_pattern = torch.rand(ir_pattern.shape).cuda()

        #test_grid = torch.zeros([1,1,1,2]).cuda()
        #test_grid[0,0,0,:] = torch.tensor([-1,-1])
        light_out = F.grid_sample(ir_pattern, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
        light_out = light_out[0,0,0,:]*attenuation_multiplier
        #print(light_out.shape)
        #assert 1==0
        surf2l_ray = (light_extrinsic[:3,3][...,None] - surface_xyz).transpose(0,1)

        

        surf2l = F.normalize(surf2l_ray,p=2.0,dim=1)

    
        t_vals = torch.linspace(
            0.0,
            1.0,
            64,
            dtype=surf2l_ray.dtype,
            device=surf2l_ray.device,
        )
        z_vals = 0 * (1.0 - t_vals) + distance * t_vals
        pts = surface_xyz[..., None, :] + surf2l[..., None, :] * z_vals[..., :, None]
        radiance_field = run_network_ir(
            model_fine,
            pts,
            ray_batch[..., -6:-3],
            getattr(options.nerf, mode).chunksize,
            encode_position_fn,
            encode_direction_fn
        )
        sigma_a = torch.nn.functional.relu(radiance_field[..., 1])
        alpha = 1.0 - torch.exp(-sigma_a * dists)
        weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)
       

        #surf2l = -l2surf
        return light_out, surf2l
    """

    def forward(self, x):

        xyz = x[..., :self.dim_xyz]
        x = self.layer1(xyz)
        for i in range(len(self.layers_xyz)):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != self.num_layers - 1
            ):
                #print("enter")
                x = torch.cat((x, xyz), dim=-1)
            #print(x.shape, len(self.layers_xyz), self.layers_xyz[i], i, self.num_layers - 1)
            x = self.relu(self.layers_xyz[i](x))
        
        out = self.fc_out(x)
        normal, brdf = out[...,:3], out[...,3:]
        normal = self.act_normal(normal)
        normal = F.normalize(normal, p=2, dim=-1)
        brdf = self.act_brdf(brdf)
        #print(brdf.shape)
        output = torch.cat((normal,brdf), dim=-1)
        #print(output.shape)
        #assert 1==0
        return output

class SGEnvironmentMap(torch.nn.Module):
    def __init__(self, num_scenes, num_lobes):
        super(SGEnvironmentMap, self).__init__()
        
        self.num_scenes = num_scenes
        self.num_lobes = num_lobes

        self.sg_params = torch.nn.parameter.Parameter(torch.zeros([num_scenes, num_lobes, 7]), requires_grad=True) 
          

    def forward(self, scene_id):
        return self.sg_params[scene_id,:,:]
        

"""
class EnvironmentModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=4,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True
    ):
        super(EnvironmentModel, self).__init__()
        self.num_layers = num_layers

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                #print(self.dim_xyz + hidden_size)
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu
"""

