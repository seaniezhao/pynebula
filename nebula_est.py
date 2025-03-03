import numpy as np
from scipy.signal import resample
from scipy.stats import norm
from scipy.optimize import minimize

def nebula_est(model, x, fs, thop=0.005):
    """
    估计基频 F0
    Args:
        model: 预训练的 GMM 模型 (dict)
        x: 输入音频信号 (numpy.ndarray)
        fs: 采样率
        thop: 时间步长 (默认 5ms)
    Returns:
        f0: 估计的基频序列
        v: 浊音(1) / 清音(0) 标志
        pv: 清音概率
        lmap: 对数似然矩阵
    """
    if fs != 8000:
        x = resample(x, int(len(x) * 8000 / fs))
    x = preprocess(x)
    fs = 8000

    nch = len(model["G"])
    taxis = np.arange(0, len(x) / fs, thop)
    naxis = np.ceil(taxis * fs).astype(int)

    fc = np.logspace(np.log10(40), np.log10(1000), nch) / fs
    fres = fc

    # 计算 SNR 和瞬时频率 IF
    SNR1, IF1, SNR2, IF2, SNR0 = estimate_ifsnr(x, naxis, fc, fres, "nuttall98", "hanning")

    eps = 1e-10
    SNR0 = 10 * np.log10(np.maximum(SNR0, eps))
    SNR1 = 10 * np.log10(np.maximum(SNR1, eps))
    SNR2 = 10 * np.log10(np.maximum(SNR2, eps))
    IF1 *= fs
    IF2 *= fs

    Lmap, f = make_likelihood_map(model, SNR1, IF1, SNR2, IF2, SNR0, fs, thop)

    LF0 = np.tile(np.arange(Lmap.shape[1]), (Lmap.shape[0], 1))
    ptrans = norm.pdf(np.arange(Lmap.shape[1]), 0, 2)
    s = viterbi_filter(LF0, np.exp(Lmap), ptrans / np.sum(ptrans))

    q, pv = detect_voicing_status(np.log(np.exp(Lmap)), s)
    v = 2 - q
    f0 = refine_f0(np.log(np.exp(Lmap)), s, f)

    return f0 * v, v, pv, Lmap


def make_likelihood_map(model, SNR1, IF1, SNR2, IF2, SNR0, fs, thop):
    """
    计算 F0 的后验概率
    """
    nch = len(model["G"])
    nf = len(model["Lcal"])
    f = np.logspace(np.log10(40), np.log10(1000), nf) / fs
    Lmap = np.full((SNR1.shape[0], nf), -100.0)

    for i in range(SNR1.shape[0]):
        Li = np.zeros((nch, nf))
        for j in range(nch):
            L = conditional_gmm(model["G"][j], f * fs, [SNR1[i, j], SNR0[i, j], SNR2[i, j], IF1[i, j], IF2[i, j]])
            L -= np.mean(L)
            Li[j, :] = L
        Lmap[i, :] = np.mean(Li, axis=0) - np.mean(model["Lcal"])
        Lmap[i, :] -= np.log(np.sum(np.exp(Lmap[i, :])))
    
    return Lmap, f


def detect_voicing_status(Lmap, s):
    """
    判断是否是清音
    """
    f0p = np.array([Lmap[i, s[i]] for i in range(len(s))])

    umean = -4.78
    ustd = 0.12

    def likelihood(x):
        return -np.mean(np.log(norm.pdf(f0p, x[0], x[1]) + norm.pdf(f0p, umean, ustd)))

    param = minimize(likelihood, [umean + 4, 1]).x
    pvoiced = norm.pdf(f0p, param[0], param[1]) / (norm.pdf(f0p, param[0], param[1]) + norm.pdf(f0p, umean, ustd))
    q = (pvoiced < 0.01).astype(int)

    return q, pvoiced


def refine_f0(Lmap, s, f):
    """
    细化基频轨迹
    """
    nf = len(f)
    Lidx = np.linspace(1, nf, nf * 20)
    Linterp = np.array([np.interp(Lidx, np.arange(1, nf + 1), row) for row in Lmap])
    sinterp = np.interp(s, np.arange(1, nf + 1), Lidx)

    f0 = np.zeros_like(s, dtype=float)
    for i in range(len(s)):
        iidx = np.clip(np.arange(int(sinterp[i]) - 10, int(sinterp[i]) + 10), 0, len(Lidx) - 1)
        f0[i] = f[np.argmax(Linterp[i, iidx])]

    return f0 * 8000