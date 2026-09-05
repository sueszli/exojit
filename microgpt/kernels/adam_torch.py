from __future__ import annotations

import torch


def adam_torch(n: int):
    @torch.compile
    def _adam(param, grad, m, v, b1, b2, eps, lr, beta1_t, beta2_t):
        inv_b1 = 1.0 - b1
        inv_b2 = 1.0 - b2
        m.mul_(b1).add_(grad, alpha=inv_b1)
        v.mul_(b2).addcmul_(grad, grad, value=inv_b2)
        m_hat = m / beta1_t
        v_hat = v / beta2_t
        param.sub_(lr * m_hat / (v_hat.sqrt() + eps))

    def wrapper(param, grad, m, v, b1, b2, eps, lr, beta1_t, beta2_t):
        _adam(
            torch.from_numpy(param),
            torch.from_numpy(grad),
            torch.from_numpy(m),
            torch.from_numpy(v),
            float(b1[0]),
            float(b2[0]),
            float(eps[0]),
            float(lr[0]),
            float(beta1_t[0]),
            float(beta2_t[0]),
        )

    return wrapper
