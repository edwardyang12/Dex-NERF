"""
Author: Isabella Liu 4/28/22
Feature: BRDF module
"""

import numpy as np

import torch
import torch.nn.functional as F


# # bug remained
# def GGX_specular(
#         normal,
#         pts2c,
#         pts2l,
#         roughness,
#         fresnel
# ):
#     L = F.normalize(pts2l, dim=-1)  # [nrays, nlights, 3]
#     V = F.normalize(pts2c, dim=-1)  # [nrays, 3]
#     H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)  # [nrays, nlights, 3]
#     N = F.normalize(normal, dim=-1)  # [nrays, 3]

#     NoV = torch.sum(V * N, dim=-1, keepdim=True)  # [nrays, 1]
#     N = N * NoV.sign()  # [nrays, 3]

#     NoL = torch.sum(N[:, None, :] * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
#     NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, 1]
#     NoH = torch.sum(N[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
#     VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]

#     alpha = roughness * roughness  # [nrays, 3]
#     alpha2 = alpha * alpha  # [nrays, 3]
#     f = NoH * NoH * (alpha2[:, None, :] - 1) + 1.0  # [nrays, nlights, 1]
#     D = alpha2[:, None, :] / (np.pi * f * f)  # [nrays, nlights, 1]

#     F0 = fresnel[:, None, :] + (1 - fresnel[:, None, :]) * torch.pow(1 - VoH, 5)  # [nrays, nlights, 3]

#     V = V_SmithGGXCorrelated(NoV[:, None, :], NoL, alpha[:, None, :])  # [nrays, nlights, 1]

#     specular = F0 * D * V  # [nrays, nlights, 3]

#     return specular

# def V_SmithGGXCorrelated(NoV, NoL, alpha):
#     alpha2 = alpha * alpha
#     GGXL = NoV * torch.sqrt((NoL * NoL * (1 - alpha2) + alpha2).clamp_(1e-12, 1))
#     GGXV = NoL * torch.sqrt((NoV * NoV * (1 - alpha2) + alpha2).clamp_(1e-12, 1))
    
#     V = 0.5 / (GGXL + GGXV)
#     return V

def specular_pipeline_render_new(pts2l, pts2c, normal, albedo=None, rough=None, f0=0.04, lambert_only=False):
    """All in the world coordinates.
    Too low roughness is OK in the forward pass, but may be numerically
    unstable in the backward pass
    pts2l: NxLx3
    pts2c: Nx3
    normal: Nx3
    albedo: Nx3
    rough: Nx1
    """
    # Normalize directions and normals
    pts2l = F.normalize(pts2l, p=2.0, dim=2)
    pts2c = F.normalize(pts2c, p=2.0, dim=1)
    normal = F.normalize(normal, p=2.0, dim=1)
    # Glossy
    h = pts2l + pts2c[:, None, :] # NxLx3
    h = F.normalize(h, p=2.0, dim=2)
    f = _get_f(pts2l, h, f0) # NxL
    alpha = rough ** 2
    d = _get_d(h, normal, alpha=alpha) # NxL
    g = _get_g(pts2c, h, normal, alpha=alpha) # NxL
    l_dot_n = torch.einsum('ijk,ik->ij', pts2l, normal)
    v_dot_n = torch.einsum('ij,ij->i', pts2c, normal)
    denom = 4 * torch.abs(l_dot_n) * torch.abs(v_dot_n)[:, None]
    microfacet = torch.divide(f * g * d, denom) # NxL
    
    brdf_glossy = microfacet[:, :, None]#.tile((1, 1, 3)) # NxLx1
    # Diffuse
    lambert = albedo / np.pi # Nx1
    brdf_diffuse = torch.broadcast_to(
        lambert[:, None, :], brdf_glossy.shape) # NxLx1
    # Mix two shaders
    if lambert_only:
        brdf = brdf_diffuse
    else:
        brdf = brdf_glossy + brdf_diffuse # TODO: energy conservation?
    return brdf # NxLx1


def _get_g(v, m, n, alpha=0.1):
    """Geometric function (GGX).
    """
    cos_theta_v = torch.einsum('ij,ij->i', n, v)
    cos_theta = torch.einsum('ijk,ik->ij', m, v)
    denom = cos_theta_v[:, None]
    div = torch.divide(cos_theta, denom)
    chi = torch.where(div > 0, 1., 0.)
    cos_theta_v_sq = torch.square(cos_theta_v)
    cos_theta_v_sq = torch.clip(cos_theta_v_sq, 0., 1.)
    denom = cos_theta_v_sq
    tan_theta_v_sq = torch.divide(1 - cos_theta_v_sq, denom)
    tan_theta_v_sq = torch.clip(tan_theta_v_sq, 0., np.inf)
    denom = 1 + torch.sqrt(1 + alpha ** 2 * tan_theta_v_sq[:, None])
    g = torch.divide(chi * 2, denom)
    return g # (n_pts, n_lights)


def _get_d(m, n, alpha=0.1):
    """Microfacet distribution (GGX).
    """
    cos_theta_m = torch.einsum('ijk,ik->ij', m, n)
    #print(m.shape, n.shape)
    chi = torch.where(cos_theta_m > 0, 1., 0.)
    cos_theta_m_sq = torch.square(cos_theta_m)
    denom = cos_theta_m_sq + 1e-10
    tan_theta_m_sq = torch.divide(1 - cos_theta_m_sq, denom)
    
    denom = np.pi * torch.square(cos_theta_m_sq) * torch.square(
        alpha ** 2 + tan_theta_m_sq)
    if not torch.all(~torch.isnan(denom)):
        nan_d = torch.isnan(torch.square(cos_theta_m_sq) * torch.square(
            alpha ** 2 + tan_theta_m_sq))
        nan_idx = (nan_d == True).nonzero(as_tuple=True)[0]
        #print(nan_idx)
        for i in nan_idx:
            print(m[i,0,:], n[i,:], tan_theta_m_sq[i,0], \
                denom[i,0])
        print("brdf: ", torch.all(~torch.isnan(torch.square(cos_theta_m_sq))).item(), torch.all(~torch.isnan(torch.square(
            alpha ** 2 + tan_theta_m_sq))).item(), torch.all(~torch.isnan(torch.square(cos_theta_m_sq) * torch.square(
            alpha ** 2 + tan_theta_m_sq))).item())
    d = torch.divide(alpha ** 2 * chi, denom)
    
    return d # (n_pts, n_lights)

def _get_f(l, m, f0):
    """Fresnel (Schlick's approximation).
    """
    cos_theta = torch.einsum('ijk,ijk->ij', l, m)
    f = f0 + (1 - f0) * (1 - cos_theta) ** 5
    return f # (n_pts, n_lights)



def specular_pipeline_render_multilight_new(
        normal,
        pts2c,
        pts2l,
        roughness,
        fresnel
):
    L = F.normalize(pts2l, dim=-1)  # [nrays, nlights, 3]
    V = F.normalize(pts2c, dim=-1)  # [nrays, 3]
    H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)  # [nrays, nlights, 3]
    N = F.normalize(normal, dim=-1)  # [nrays, 3]

    NoV = torch.sum(V * N, dim=-1, keepdim=True)  # [nrays, 1]
    N = N * NoV.sign()  # [nrays, 3]

    NoL = torch.sum(N[:, None, :] * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1] TODO check broadcast
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, 1]
    NoH = torch.sum(N[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
    VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]

    alpha = roughness * roughness  # [nrays, 1]
    alpha2 = alpha * alpha  # [nrays, 1]
    k = (alpha + 2 * roughness + 1.0) / 8.0
    FMi = ((-5.55473) * VoH - 6.98316) * VoH
    frac0 = fresnel[:, None, :] + (1 - fresnel[:, None, :]) * torch.pow(2.0, FMi)  # [nrays, nlights, 1]
    
    frac = frac0 * alpha2[:, None, :]  # [nrays, 1]
    nom0 = NoH * NoH * (alpha2[:, None, :] - 1) + 1

    nom1 = NoV * (1 - k) + k
    nom2 = NoL * (1 - k[:, None, :]) + k[:, None, :]
    nom = (4 * np.pi * nom0 * nom0 * nom1[:, None, :] * nom2).clamp_(1e-6, 4 * np.pi)
    spec = frac / nom
    return spec


def specular_pipeline_render_multilight(
        normal,
        pts2c,
        pts2l,
        roughness,
        fresnel,
):
    # ref-nerf - frequency - monte carlo - integeral
    """
    Args:
        ray_dir: :math:`(Rays/1,Samples/1,3)
        ray_feature: :math:`(N,Rays,Samples,features)`
        lightdir: :math:`(Npt,nlights,3)
    """
    """
    New Args:
        ray_dir: :math:`(nrays,3)
        lightdir: :math:`(nrays,nlights,3)
    Return
         specular: (nrays,nlights,3)
    """

    L = F.normalize(pts2l, dim=-1)  # [nrays, nlights, 3]

    V = F.normalize(pts2c, dim=-1)  # [nrays, 3]
    H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)  # [nrays, nlights, 3]
    N = F.normalize(normal, dim=-1)  # [nrays, 3]
    # NoV = torch.sum(V*N, dim=-1, keepdim=True)
    # N = N * NoV.sign()

    alpha = roughness * roughness

    def cal_spec_F():
        # specular F
        VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True)  # (npt,nlight)
        FMi = ((-5.55473) * VoH - 6.98316) * VoH  # interpolation
        frac0 = fresnel[:, None, :] + (1 - fresnel[:, None, :]) * torch.pow(2.0,
                                                                            FMi)  # or  (1-fresnel[:,None,:])*torch.pow(VoH,5)
        return frac0  # (npt,nlights,3)

    def cal_spec_D():
        # specular D
        NoH = torch.sum(N[:, None, :] * H)
        alpha2 = alpha * alpha
        nom0 = NoH * NoH * (alpha2 - 1) + 1  # nom of D
        return alpha2, nom0 * nom0  # (npt,1),(npt,nlights)

    def cal_spec_G():
        # specular G
        k = (alpha + 2 * roughness + 1.0) / 8.0
        NoL = torch.sum(N[:, None, :] * L, dim=-1)
        NoV = torch.sum(N * V, dim=-1, keepdim=True)
        res = (NoV * (1 - k) + k) * (NoL * (1 - k) + k)
        return res  # (npt,nlights)

    frac, nom02 = cal_spec_D()
    nom = (4 * np.pi * nom02 * cal_spec_G()).clamp_(1e-6, 4 * np.pi)
    spec = frac[:, None, :] * cal_spec_F() / nom[:, :, None]
    return spec


def specular_pipeline_render_multilight_old(
        normal,
        pts2c,
        pts2l,
        roughness,
        fresnel,
):
    # ref-nerf - frequency - monte carlo - integeral
    """
    Args:
        ray_dir: :math:`(Rays/1,Samples/1,3)
        ray_feature: :math:`(N,Rays,Samples,features)`
        lightdir: :math:`(Npt,nlights,3)
    """
    """
    New Args:
        ray_dir: :math:`(nrays,3)
        lightdir: :math:`(nrays,nlights,3)
    Return
         specular: (nrays,nlights,3)
    """

    L = F.normalize(pts2l, dim=-1)

    V = F.normalize(pts2c, dim=-1)
    H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)  # (npt,nlight,3)
    N = F.normalize(normal, dim=-1)
    NoV = torch.sum(V * N, dim=-1, keepdim=True)
    N = N * NoV.sign()

    alpha = roughness * roughness

    def cal_spec_F():
        # specular F
        VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # (npt,nlight)
        FMi = ((-5.55473) * VoH - 5.98316) * VoH  # interpolation
        frac0 = fresnel[:, None, :] + (1 - fresnel[:, None, :]) * torch.pow(2.0, FMi)
        return frac0  # (npt,nlights,3)

    def cal_spec_D():
        # specular D
        NoH = torch.sum(N[:, None, :] * H).clamp_(1e-6, 1)
        alpha2 = alpha * alpha
        nom0 = NoH * NoH * (alpha2 - 1) + 1  # nom of D
        return alpha2, nom0 * nom0  # (npt,1),(npt,nlights)

    def cal_spec_G():
        # specular G
        k = (alpha + 2 * roughness + 1.0) / 8.0  # [bs, ]
        NoL = torch.sum(N[:, None, :] * L, dim=-1).clamp_(1e-6, 1)  # [bs, nlights]
        NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [bs, 1]
        res = (NoV * (1 - k) + k) * (NoL * (1 - k) + k)
        return res  # (npt,nlights)

    frac, nom02 = cal_spec_D()
    nom = (4 * np.pi * nom02 * cal_spec_G()).clamp_(1e-6, 4 * np.pi)
    spec = frac[:, None, :] * cal_spec_F() / nom[:, :, None]
    return spec
