import numpy as np
from numpy import pi, ceil, floor, exp, cos
from scipy.signal import fftconvolve
from ifdetect import ifdetect

def getwcoef(wtype):
    """
    获取指定窗口类型的权重系数。
    """
    if wtype == 'nuttall83':
        return np.array([0.338946, 0.481973, 0.161054, 0.018027])
    elif wtype == 'nuttall98':
        return np.array([0.3635819, 0.4891775, 0.1365995, 0.0106411])
    elif wtype == 'nuttall93':
        return np.array([0.355768, 0.487396, 0.144232, 0.012604])
    elif wtype == 'nuttall64':
        return np.array([0.40897, 0.5, 0.09103, 0])
    elif wtype == 'hanning':
        return np.array([0.5, 0.5, 0, 0])
    elif wtype == 'hamming':
        return np.array([0.54, 0.46, 0, 0])
    elif wtype == 'blackman':
        return np.array([0.42, 0.5, 0.08, 0])
    elif wtype == 'blackman-harris':
        return np.array([0.4243801, 0.4973406, 0.0782793, 0])
    elif wtype == 'boxcar':
        return np.array([1, 0, 0, 0])
    else:
        raise ValueError(f"Unknown window type: {wtype}")

def fftconv(x, h):
    """
    利用 FFT 进行卷积，模拟 MATLAB 中的 fftconv 函数。
    """
    return fftconvolve(x, h, mode='full')

def x2as(x, h, ws):
    """
    将信号 x 通过两次滤波和能量计算，得到归一化的能量特征（as）。
    
    对应 MATLAB 代码：
      y1 = fftconv(x, h)(nw_2+1:end-nw+nw_2+1);
      y1 ./= abs(y1) + eps;
      y2 = fftconv(y1, h)(nw_2+1:end-nw+nw_2+1);
      r = y1 - y2;
      a = abs(r) .^ 2;
      as = fftconv(a, ws)(nw_2+1:end-nw+nw_2+1);
      as(1:nw) = as(nw);
      as(end-nw:end) = as(end-nw);
    """
    nw = len(h)
    nw_2 = int(np.floor(nw / 2))
    
    # 第一次卷积
    y1_full = fftconv(x, h)
    # MATLAB 索引： (nw_2+1):(end-nw+nw_2+1)
    start = nw_2
    end = len(y1_full) - nw + nw_2 + 1
    y1 = y1_full[start:end]
    
    eps_val = np.finfo(float).eps
    y1 = y1 / (np.abs(y1) + eps_val)
    
    # 第二次卷积
    y2_full = fftconv(y1, h)
    start2 = nw_2
    end2 = len(y2_full) - nw + nw_2 + 1
    y2 = y2_full[start2:end2]
    
    r = y1 - y2
    a = np.abs(r) ** 2
    
    # 第三次卷积（平滑）
    as_full = fftconv(a, ws)
    start3 = nw_2
    end3 = len(as_full) - nw + nw_2 + 1
    a_s = as_full[start3:end3]
    
    # 将边缘处的值设为边缘内的稳定值
    a_s[:nw] = a_s[nw-1]
    a_s[-nw:] = a_s[-nw]
    
    return a_s

def estimate_ifsnr(x, naxis, fc, fres, wtype='nuttall83', stype='nuttall83'):
    """
    估计信号的瞬时频率 (IF) 和信噪比 (SNR) 特征。
    
    参数：
      x     : 输入信号（1D numpy 数组）
      naxis : 分析时刻的索引（1D 数组，注意：已调整为 0-indexed）
      fc    : 中心频率（数组，作为采样率的比例）
      fres  : 带宽（数组，作为采样率的比例）
      wtype : 分析窗口类型（默认 'nuttall83'）
      stype : 平滑窗口类型（默认 'nuttall83'）
      
    返回：
      SNR1, IF1, SNR2, IF2, SNR0
      分别为：
        - SNR1 : 在中心频率处的 SNR 特征矩阵（每列对应一个频带）
        - IF1  : 在中心频率处的瞬时频率特征矩阵
        - SNR2 : 在双倍中心频率处的 SNR 特征矩阵
        - IF2  : 在双倍中心频率处的瞬时频率特征矩阵
        - SNR0 : 由 SNR1 的特定列构成的特征矩阵（用于后续处理）
    """
    nch = len(fc)
    ntime = len(naxis)
    SNR1 = np.zeros((ntime, nch))
    SNR2 = np.zeros((ntime, nch))
    IF1  = np.zeros((ntime, nch))
    IF2  = np.zeros((ntime, nch))
    
    # 加载窗口系数
    aw = getwcoef(wtype)
    aws = getwcoef(stype)
    
    # 对于每个频带
    for i in range(nch):
        # 计算窗口长度
        nw = int(np.ceil(4 / fres[i]))
        nw_2 = int(np.floor(nw / 2))
        omegaw = 2 * np.pi * fres[i] / 4
        
        w = np.zeros(nw)
        ws = np.zeros(nw)
        # 累加加权余弦窗（j=1,...,4）
        for j in range(1, 5):
            jw = np.cos((j - 1) * omegaw * (np.arange(nw) - nw_2))
            w += aw[j-1] * jw
            ws += aws[j-1] * jw
        w = w / np.sum(w)
        ws = ws / np.sum(ws)
        
        omegah = 2 * np.pi * fc[i]
        h = w * np.exp(1j * omegah * (np.arange(nw) - nw_2))
        h2 = w * np.exp(1j * omegah * (np.arange(nw) - nw_2) * 2)
        
        # 计算 SNR 特征
        SNR1[:, i] = x2as(x, h, ws)[naxis]
        SNR2[:, i] = x2as(x, h2, ws)[naxis]
        
        # 计算瞬时频率特征（IF）
        fi = ifdetect(x, naxis, fc[i], fres[i])
        fi = np.clip(fi, fc[i] * 0.5, fc[i] * 1.5)
        IF1[:, i] = fi
        
        fi2 = ifdetect(x, naxis, fc[i] * 2, fres[i])
        fi2 = np.clip(fi2, fc[i] * 1.5, fc[i] * 2.5)
        IF2[:, i] = fi2
        
    # 构造 SNR0, MATLAB: [SNR1(:, 8:14) SNR1(:, 1:end-7)]
    # 注意 MATLAB 索引 8:14 对应 Python 中索引 7:14，
    # MATLAB 1:end-7 对应 Python 0:(end-7)
    SNR0 = np.hstack((SNR1[:, 7:14], SNR1[:, 0:-7]))
    
    return SNR1, IF1, SNR2, IF2, SNR0

if __name__ == "__main__":
    # 示例：构造伪数据测试函数
    # 生成随机信号
    x = np.random.randn(10000)
    # 假设分析时刻的索引（注意：MATLAB 中索引从1开始，Python 从0开始，这里假设已经调整）
    naxis = np.arange(100, 9900, 100)
    # 设定中心频率（作为采样率的比例）和带宽
    fc = np.linspace(0.05, 0.4, 20)    # 20个频带
    fres = np.linspace(0.01, 0.05, 20)
    
    SNR1, IF1, SNR2, IF2, SNR0 = estimate_ifsnr(x, naxis, fc, fres)
    print("SNR1 shape:", SNR1.shape)
    print("IF1 shape:", IF1.shape)
    print("SNR0 shape:", SNR0.shape)