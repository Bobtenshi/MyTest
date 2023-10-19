import numpy as np
import numpy.linalg as LA


def cost_function(X, R, W):
    """_summary_

    Args:
        X (_type_): I,J,M
        R (_type_): M,I,J
        W (_type_): I,M,M

    Returns:
        _type_: _description_
    """
    x = X.transpose([0, 2, 1])
    y = W @ x
    # (n_freq, n_src, n_frame )
    est = y.transpose([2, 0, 1])  # (t,f,m )

    y_power = np.square(np.abs(est))
    src_var = R.transpose([2, 1, 0])  # (t,f,m )

    # (n_freq,)
    target_loss = -2 * np.linalg.slogdet(W)[1]
    # (n_frame, n_freq)
    demix_loss = np.sum(y_power / src_var + np.log(src_var), axis=2)
    cost = np.sum(demix_loss + target_loss[None, :])

    return cost


def cost_function_old(P, Rc, W, I, J):
    """
    P,R (m,req,time)
    """

    x = np.abs(LA.det(W))
    x = np.maximum(x, 1e-15)
    B = np.log(x)

    # cost = -2 * J * B.sum() + (P / R + np.log(R)).sum()
    cost = -2 * B.sum() + ((P / R + np.log(R)).mean(axis=(0, 2))).sum()
    # cost = -2 * B.mean() + (P / R + np.log(R)).mean()
    return cost
