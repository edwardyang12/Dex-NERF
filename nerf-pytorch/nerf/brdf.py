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
