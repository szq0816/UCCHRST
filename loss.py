import torch
import torch.nn.functional as F

def Hinge_loss(f, psd_label, margin):
    N = f.size(0)
    f = F.normalize(f, p=2, dim=-1)
    psd_label = F.normalize(psd_label, p=2, dim=-1)
    l_f_product = psd_label.matmul(f.t())
    l_f_diag = torch.diag(l_f_product).unsqueeze(0)
    l_f_d = l_f_product - l_f_diag
    L = (torch.sum(F.relu(l_f_d + margin)) - N*margin) / (N**2 - N)

    return L