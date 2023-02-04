import torch
import numpy as np
from nerf import math_utils

class SgRenderer(torch.nn.Module):
    def __init__ (
        self,
        eval_background: bool = False,
        compress_sharpness: bool = False,
        compress_amplitude: bool = False,
    ):
        super().__init__()
        self.eval_background = eval_background
        self.compress_sharpness = compress_sharpness
        self.compress_amplitude = compress_amplitude
    
    def forward(self,
        sg_illuminations,
        basecolor,
        metallic,
        roughness,
        normal,
        alpha,
        view_dir,
    ):
        # basecolor is input
        # metallic is another input
        lin_basecolor = math_utils.srgb_to_linear(basecolor)
        diffuse = lin_basecolor * (1 - metallic)
        specular = math_utils.mix(
                    torch.ones_like(lin_basecolor) * 0.04, lin_basecolor, metallic
            )
        normal = torch.where(normal == torch.zeros_like(normal), view_dir, normal)

        diffuse = torch.unsqueeze(diffuse, 1)
        specular = torch.unsqueeze(specular, 1)
        roughness = torch.unsqueeze(roughness, 1)
        normal = torch.unsqueeze(math_utils.normalize(normal), 1)
        view_dir = torch.unsqueeze(math_utils.normalize(view_dir), 1)

        env = None 
        if self.eval_background:
            # Evaluate everything for the environment
            env = self._sg_evaluate(sg_illuminations, view_dir)
            # And sum all contributions
            env = torch.sum(env, 1)
        brdf = self._brdf_eval(
                sg_illuminations, diffuse, specular, roughness, normal, view_dir,
            )
        brdf = torch.sum(brdf, 1)

        m = torch.nn.ReLU()
        if self.eval_background:
            if len(alpha.shape) == 1:
                alpha = torch.unsqueeze(alpha, 1)
            alpha = torch.clip(alpha, 0, 1)

            return m(brdf * alpha + env * (1 - alpha))
        else:
            return m(brdf)

    def _sg_evaluate(self, sg, d):
        s_amplitude, s_axis, s_sharpness = self._extract_sg_components(sg)

        cosAngle = math_utils.dot(d, s_axis)
        return s_amplitude * math_utils.safe_exp(s_sharpness * (cosAngle - 1.0))

    def _extract_sg_components(self, sg):
        s_amplitude = (
            math_utils.safe_exp(sg[..., 0:3])
            if self.compress_amplitude
            else sg[..., 0:3]
        )
        s_axis = sg[..., 3:6]
        s_sharpness = (
            math_utils.safe_exp(sg[..., 6:7])
            if self.compress_amplitude
            else sg[..., 6:7]
        )

        return (
            torch.abs(s_amplitude),
            math_utils.normalize(s_axis),
            math_utils.saturate(s_sharpness, 0.5, 30),
        )

    def _brdf_eval(
        self,
        sg_illuminations,
        diffuse,
        specular,
        roughness,
        normal,
        view_dir,
    ):
        v = view_dir
        diff = diffuse
        spec = specular
        norm = normal
        rogh = roughness

        ndf = self._distribution_term(norm, rogh)

        warped_ndf = self._sg_warp_distribution(ndf, v)
        _, warpDir, _ = self._extract_sg_components(warped_ndf)

        ndl = math_utils.saturate(math_utils.dot(norm, warpDir))
        ndv = math_utils.saturate(math_utils.dot(norm, v))
        h = math_utils.normalize(warpDir + v)
        ldh = math_utils.saturate(math_utils.dot(warpDir, h))

        diffuse_eval = self._evaluate_diffuse(sg_illuminations, diff, norm)  # * ndl
        specular_eval = self._evaluate_specular(
            sg_illuminations, spec, rogh, warped_ndf, ndl, ndv, ldh
        )
        return diffuse_eval + specular_eval

    def _distribution_term(self, d, roughness):
        a2 = math_utils.saturate(roughness * roughness, 1e-3)

        ret = self._stack_sg_components(
            math_utils.to_vec3(torch.reciprocal(np.pi * a2)),
            d,
            2.0 / torch.maximum(a2, torch.tensor(1e-6)),
        )

        return ret

    def _stack_sg_components(self, amplitude, axis, sharpness):
        return torch.cat(
            [
                math_utils.safe_log(amplitude)
                if self.compress_amplitude
                else amplitude,
                axis,
                math_utils.safe_log(math_utils.saturate(sharpness, 0.5, 30))
                if self.compress_sharpness
                else sharpness,
            ],
            -1,
        )
    
    def _sg_warp_distribution(self, ndfs, v):
        ndf_amplitude, ndf_axis, ndf_sharpness = self._extract_sg_components(ndfs)

        ret = torch.cat(
            [
                ndf_amplitude,
                math_utils.reflect(-v, ndf_axis),
                torch.divide(
                    ndf_sharpness,
                    (4.0 * math_utils.saturate(math_utils.dot(ndf_axis, v), 1e-4)),
                ),
            ],
            -1,
        )

        return ret

    def _extract_sg_components(self, sg):
        s_amplitude = (
            math_utils.safe_exp(sg[..., 0:3])
            if self.compress_amplitude
            else sg[..., 0:3]
        )
        s_axis = sg[..., 3:6]
        s_sharpness = (
            math_utils.safe_exp(sg[..., 6:7])
            if self.compress_amplitude
            else sg[..., 6:7]
        )

        return (
            torch.abs(s_amplitude),
            math_utils.normalize(s_axis),
            math_utils.saturate(s_sharpness, 0.5, 30),
        )

    def _evaluate_diffuse(
        self, sg_illuminations, diffuse, normal
    ):
        diff = diffuse / np.pi

        _, s_axis, s_sharpness = self._extract_sg_components(sg_illuminations)
        mudn = math_utils.saturate(math_utils.dot(s_axis, normal))

        c0 = 0.36
        c1 = 1.0 / (4.0 * c0)

        eml = math_utils.safe_exp(-s_sharpness)
        em2l = eml * eml
        rl = torch.reciprocal(s_sharpness)

        scale = 1.0 + 2.0 * em2l - rl
        bias = (eml - em2l) * rl - em2l

        x = math_utils.safe_sqrt(1.0 - scale)
        x0 = c0 * mudn
        x1 = c1 * x

        n = x0 + x1

        y_cond = torch.le(torch.abs(x0), x1)
        y_true = n * (n / torch.maximum(x, torch.tensor(1e-6)))
        y_false = mudn
        y = torch.where(y_cond, y_true, y_false)

        res = scale * y + bias

        res = res * self._sg_integral(sg_illuminations) * diff

        return res

    def _evaluate_specular(
        self,
        sg_illuminations,
        specular,
        roughness,
        warped_ndf,
        ndl,
        ndv,
        ldh,
    ):
        a2 = math_utils.saturate(roughness * roughness, 1e-3)
        D = self._sg_inner_product(warped_ndf, sg_illuminations)
        G = self._ggx(a2, ndl) * self._ggx(a2, ndv)

        powTerm = torch.pow(1.0 - ldh, 5)
        F = specular + (1.0 - specular) * powTerm

        output = D * G * F * ndl

        m = torch.nn.ReLU()
        return m(output)


    def _sg_integral(self, sg):
        s_amplitude, _, s_sharpness = self._extract_sg_components(sg)

        expTerm = 1.0 - math_utils.safe_exp(-2.0 * s_sharpness)
        return 2 * np.pi * torch.divide(s_amplitude, s_sharpness) * expTerm

    def _sg_inner_product(self, sg1, sg2):
        s1_amplitude, s1_axis, s1_sharpness = self._extract_sg_components(sg1)
        s2_amplitude, s2_axis, s2_sharpness = self._extract_sg_components(sg2)

        umLength = math_utils.magnitude(
            s1_sharpness * s1_axis + s2_sharpness * s2_axis
        )
        expo = (
            math_utils.safe_exp(umLength - s1_sharpness - s2_sharpness)
            * s1_amplitude
            * s2_amplitude
        )

        other = 1.0 - math_utils.safe_exp(-2.0 * umLength)

        return torch.divide(2.0 * np.pi * expo * other, umLength)

    def _ggx(self, a2, ndx):
        return torch.reciprocal(
            ndx + math_utils.safe_sqrt(a2 + (1 - a2) * ndx * ndx))

    def _sg_evaluate(self, sg, d):
        s_amplitude, s_axis, s_sharpness = self._extract_sg_components(sg)

        cosAngle = math_utils.dot(d, s_axis)
        return s_amplitude * math_utils.safe_exp(s_sharpness * (cosAngle - 1.0))

if __name__ == "__main__":
    render = SgRenderer(True, True, True)
    a = torch.tensor(np.array([[ 0.72247012, -0.86145812,  0.85896839],
       [ 0.77770441, -0.24162251, -1.74438422],
       [ 1.1377586 ,  0.13425733, -0.85531416]]))
    b = torch.tensor(np.array([[[-0.94271081,  0.18759034,  0.86982421, -0.18818365,
          0.61540824, -1.84733081,  0.43973863],
        [-0.28000091,  1.40378771,  0.51750983, -1.77231995,
          0.41252113, -1.80419495, -3.19161534],
        [ 1.0779477 ,  1.18253241,  0.39783513,  0.80997685,
         -0.61990218, -0.65476139, -1.61334007],
        [-0.95532886,  0.38195151, -2.46179194,  0.40461472,
         -1.11370041, -0.78094197,  1.37512742],
        [ 0.22168887,  0.2145094 ,  1.77143894, -0.46331855,
          1.24919736, -0.20203875, -0.23584248],
        [ 0.27967487, -1.49199174, -0.3003819 ,  1.24065207,
          0.35452987,  0.65793311, -0.41719085],
        [ 0.63117398, -0.18708784,  0.61476719, -1.43966991,
         -0.1973517 ,  0.42366299,  0.12702552],
        [ 1.14205236,  0.89285348, -0.61652231,  0.32499764,
          0.52459015,  0.53975197, -0.82008253],
        [-1.57895935,  0.75977238, -0.25729153, -0.96820909,
          1.00336055, -1.38125658, -0.52822184],
        [-0.52822774, -0.37195686,  0.58074479, -0.01282806,
          1.86310017, -0.51489432, -1.5080561 ],
        [-0.51115027, -0.78090194,  0.04585303,  0.16140357,
         -0.04817827, -0.86365369,  0.64344935],
        [-1.72615271,  0.02382109, -2.15765199,  0.86169384,
         -0.981989  , -1.38237457,  0.43390144],
        [ 1.34244304, -0.5738464 ,  1.97536162,  1.13657098,
          0.23217602,  1.23508587, -0.88129357],
        [ 1.38173721,  0.26190792, -0.75089959,  0.65054388,
         -0.37870563, -0.1246181 ,  0.70080653],
        [ 0.3312455 ,  0.26268453, -1.24736935,  0.80776042,
          0.35490716, -0.06670291, -1.07947891],
        [-0.90262347,  0.61595701, -1.68541624, -0.19920794,
         -0.98259543,  0.67750277, -1.24394443],
        [ 0.24631536,  0.24759469,  0.57424144,  0.29292993,
         -1.45296997, -0.97149331, -3.08977705],
        [ 1.00102106,  0.24840012, -2.08014287,  0.11371739,
          1.33156436, -0.11078625,  0.12446222],
        [ 1.02986193, -0.33855397, -0.11232679, -0.46864761,
         -0.23486365, -0.68522427,  0.52551011],
        [ 0.55731136,  0.49123552, -0.81200703,  1.32531313,
         -0.38947371, -0.03763758, -0.42203947],
        [ 0.33828652,  0.20967499,  0.42211563, -1.5412268 ,
         -0.42510433, -0.8355421 , -1.05056309],
        [-0.80663323,  0.85202479, -0.77575392, -0.19436219,
          0.82700843, -1.04936681, -0.78993511],
        [ 1.22077713, -1.08019284,  0.07153447, -0.26995556,
         -1.19585732, -0.77317055,  1.11639269],
        [ 1.80649022,  0.35256355, -0.96615839,  1.99424972,
          0.2238658 ,  0.75227578, -1.80881099]]]))
    sgs_illumination = b #(1,24,7)
    basecolor = a # (batch, 3)
    metallic = a
    roughness = a
    normal = a
    alpha = a
    ray_directions = a
    output = render(
                sg_illuminations=sgs_illumination,
                basecolor=basecolor,
                metallic=metallic,
                roughness=roughness,
                normal=normal,
                alpha=alpha,
                view_dir=ray_directions,
            )
    print(output)