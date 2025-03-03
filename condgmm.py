import numpy as np
 


def conditional_gmm(gmm, f_values, x_values):
    """
    计算目标变量 f 在 GMM 下的条件概率。
    
    参数:
    - gmm: 训练好的 GMM 模型
    - f_values: (nf,) 目标变量 (F0) 可能取值
    - x_values: (5,) 观测变量 (SNR1, SNR0, SNR2, IF1, IF2)

    返回:
    - L: (nf,) 对数似然数组
    """
    means = gmm.means_
    covs = gmm.covariances_
    weights = gmm.weights_

    x_dim = 5  # 观测变量维度
    f_dim = 1  # 目标变量维度
    
    L = np.zeros(len(f_values))  # 存储对数似然

    for k in range(gmm.n_components):
        mu_x = means[k, :x_dim]
        mu_f = means[k, x_dim]
        cov_xx = covs[k, :x_dim, :x_dim]
        cov_ff = covs[k, x_dim, x_dim]
        cov_fx = covs[k, x_dim, :x_dim]

        # 计算条件均值和方差
        inv_cov_xx = np.linalg.inv(cov_xx)
        mu_f_given_x = mu_f + cov_fx @ inv_cov_xx @ (x_values - mu_x)
        sigma_f_given_x = cov_ff - cov_fx @ inv_cov_xx @ cov_fx.T

        # 计算 P(f | x) 的概率密度
        p_f_given_x = (1 / np.sqrt(2 * np.pi * sigma_f_given_x)) * np.exp(-0.5 * (f_values - mu_f_given_x)**2 / sigma_f_given_x)

        # 混合权重
        L += weights[k] * p_f_given_x

    return np.log(L + 1e-10)  # 避免 log(0) 问题