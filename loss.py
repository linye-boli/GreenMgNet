import torch
import torch.nn.functional as F

def toep_gradient_loss(A, h, max_grad=3):
    n = A.shape[-1]
    A = A[0,0]
    Agrad = torch.gradient(A, spacing=h)[0]
    loss = F.relu(Agrad.abs() - max_grad).mean()
    return loss