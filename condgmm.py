import numpy as np

def lse(a, b):
    """
    数值稳定地计算 log(exp(a)+exp(b))
    """
    maxab = a if a > b else b
    minab = b if a > b else a
    if maxab - minab > 15:
        return maxab
    return np.log(1.0 + np.exp(minab - maxab)) + maxab

def condgmm(G, f, x, stdfloor=0.0):
    """
    根据条件 GMM 计算候选频率点的对数似然评分。
    
    参数：
      G       : 列表，每个元素是字典，包含混合成分的参数：
                'mu' (均值向量, 长度 nx+1),
                'w' (混合权重),
                'inv11' (一维数组，尺寸 nx*nx，按列优先排列),
                'det11' (标量，行列式),
                's21s11i' (一维数组, 长度 nx),
                'condsigma' (标量)
      f       : numpy 数组，候选频率，长度 nf
      x       : numpy 数组，输入特征向量，长度 nx
      stdfloor: 标量，防止条件标准差过小（默认 0）
      
    返回：
      L : numpy 数组，长度 nf，候选频率点上计算得到的对数似然（log likelihood）。
    """
    nx = len(x)
    nf = len(f)
    nmix = len(G)
    f0_mu    = np.zeros(nmix)
    f0_sigma = np.zeros(nmix)
    f0_lL    = np.zeros(nmix)
    f0_w     = np.zeros(nmix)
    dx       = np.zeros(nx)
    lwsum = -1e10

    # 对每个混合成分进行处理
    for i in range(nmix):
        comp = G[i]
        # joint_mu: 假设是一个至少长度 nx+1 的数组
        joint_mu = np.array(comp["mu"]).flatten()
        joint_w  = comp["w"]
        inv11    = np.array(comp["inv11"]).flatten()
        det11    = abs(comp["det11"])
        S21_S11inv = np.array(comp["s21s11i"]).flatten()
        cond_sigma = comp["condsigma"]
        
        # MATLAB 中 f0_mu[i] = joint_mu(nx+1)，对应 Python 索引 nx
        f0_mu[i] = joint_mu[nx]
        # 防止条件标准差过小
        f0_sigma[i] = max(stdfloor * stdfloor, cond_sigma)
        # 计算 dx 和修正 f0_mu
        for k in range(nx):
            dx[k] = x[k] - joint_mu[k]
            f0_mu[i] += S21_S11inv[k] * dx[k]
        
        # 计算 dxS11invdx = dx' * inv11 * dx
        # 将 inv11 按列优先重塑成矩阵 (nx, nx)
        inv11_mat = np.array(inv11).reshape((nx, nx), order='F')
        dxS11invdx = 0.0
        for j in range(nx):
            dx_S11inv = np.sum(dx * inv11_mat[:, j])
            dxS11invdx += dx_S11inv * dx[j]
        
        f0_lL[i] = (-0.5 * dxS11invdx -
                    (nx - 1) * np.log(2.0 * np.pi) -
                    0.5 * np.log(det11) +
                    np.log(joint_w))
        lwsum = lse(lwsum, f0_lL[i])
    
    # 调整混合权重
    wsum = 0.0
    for i in range(nmix):
        lw = f0_lL[i] - lwsum
        if lw < -10:
            lw = -10
        f0_w[i] = np.exp(lw)
        wsum += f0_w[i]
    for i in range(nmix):
        f0_w[i] /= wsum
    
    # 计算候选频率点上的条件似然
    L = np.zeros(nf)
    for i in range(nf):
        p = 0.0
        for j in range(nmix):
            d = (f[i] - f0_mu[j])**2 / f0_sigma[j]
            p += f0_w[j] * np.exp(-0.5 * d) / np.sqrt(2.0 * np.pi * f0_sigma[j])
        L[i] = np.log(p)
    
    return L

# 示例用法：
if __name__ == '__main__':
    # 假设我们有 3 个混合成分
    G = []
    nmix = 3
    nx = 5   # 假设 x 长度为 5（实际应用中维度可能更高）
    for i in range(nmix):
        comp = {
            "mu": np.concatenate((np.linspace(0.1, 0.5, nx), [0.3])),  # 长度 nx+1
            "w": 0.3 + 0.1 * i,  # 混合权重
            "inv11": np.eye(nx).flatten(order='F'),  # 单位矩阵（简单示例）
            "det11": 1.0,
            "s21s11i": np.ones(nx) * 0.05,
            "condsigma": 0.2 + 0.05 * i
        }
        G.append(comp)
    # 候选频率点：假设有 10 个频率点
    f = np.linspace(0.05, 0.15, 10)
    # 输入特征向量 x，长度 nx
    x = np.linspace(0.2, 0.4, nx)
    # stdfloor 可选（例如设为 0.1）
    stdfloor = 0.1

    L = condgmm(G, f, x, stdfloor)
    print("候选频率上的对数似然向量 L:")
    print(L)