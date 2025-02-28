import numpy as np

def fetch_frame(x, center, nh):
    """
    从信号 x 中提取长度 nh 的帧，中心位置为 center。
    如果超出信号边界，则用零填充。
    """
    half = nh // 2
    start = center - half
    end = start + nh
    frame = np.zeros(nh, dtype=x.dtype)
    for i in range(nh):
        idx = start + i
        if 0 <= idx < len(x):
            frame[i] = x[idx]
    return frame

class IfDetector:
    def __init__(self, fc, fres):
        """
        根据 cig_create_ifdetector 的思路构造 ifdetector 对象
        参数：
          fc   : 中心频率（标量）
          fres : 带宽（标量）
        """
        self.fc = fc
        self.fres = fres
        # 窗口长度 nh：取 ceil(4 / fres)
        self.nh = int(np.ceil(4 / fres))
        nh = self.nh

        # 分配滤波器系数数组
        self.hr = np.zeros(nh)
        self.hi = np.zeros(nh)
        self.hdr = np.zeros(nh)
        self.hdi = np.zeros(nh)

        omega = 2 * np.pi * fc
        omegaw = 2 * np.pi / nh

        # a 数组系数，固定为 [0.338946, 0.481973, 0.161054, 0.018027]
        a = np.array([0.338946, 0.481973, 0.161054, 0.018027])
        # 先计算 hr 和 hdr
        for i in range(nh):
            for k in range(4):
                self.hr[i] += a[k] * np.cos(k * omegaw * (i - nh/2))
                self.hdr[i] += -omegaw * k * a[k] * np.sin(k * omegaw * (i - nh/2))
        # 再计算 hi、hdi，同时修正 hr 和 hdr
        for i in range(nh):
            sini = np.sin(omega * (i - nh/2))
            cosi = np.cos(omega * (i - nh/2))
            w = self.hr[i]
            wd = self.hdr[i]
            self.hr[i] = w * cosi
            self.hi[i] = w * sini
            self.hdi[i] = omega * w * cosi + wd * sini
            self.hdr[i] = wd * cosi - omega * w * sini

    def estimate(self, frame):
        """
        对单帧信号 frame 进行瞬时频率估计
        取 frame 中心 nh 个样本（这里如果 frame 长度正好为 nh，则 n0=0），
        然后计算:
          yr = Σ hr[i]*frame[n0+i]
          yi = Σ hi[i]*frame[n0+i]
          ydr = Σ hdr[i]*frame[n0+i]
          ydi = Σ hdi[i]*frame[n0+i]
        最后返回 f = (yr*ydi - yi*ydr)/(yr²+yi²) / (2π)
        """
        nx = len(frame)
        if nx < self.nh:
            return 0.0
        nh = self.nh
        n0 = int(nx/2 - nh/2)
        yr = 0.0
        yi = 0.0
        ydr = 0.0
        ydi = 0.0
        for i in range(nh):
            yr += self.hr[i] * frame[n0 + i]
            yi += self.hi[i] * frame[n0 + i]
            ydr += self.hdr[i] * frame[n0 + i]
            ydi += self.hdi[i] * frame[n0 + i]
        denom = yr * yr + yi * yi
        if denom == 0:
            return 0.0
        return (yr * ydi - yi * ydr) / denom / (2 * np.pi)

def ifdetect(x, naxis, fc, fres):
    """
    根据 ciglet 的 ifdetect 实现瞬时频率检测。
    
    参数：
      x     : 1D numpy 数组，输入信号
      naxis : 1D 数组，分析时刻的索引（整型，0-indexed）
      fc    : 中心频率（标量）
      fres  : 带宽（标量）
      
    返回：
      f     : 1D numpy 数组，每个分析时刻对应的瞬时频率估计
    """
    detector = IfDetector(fc, fres)
    f = np.zeros(len(naxis))
    for i, center in enumerate(naxis):
        # 提取长度为 detector.nh 的帧（若超出边界则零填充）
        frame = fetch_frame(x, int(center), detector.nh)
        f[i] = detector.estimate(frame)
    return f

# 示例测试
if __name__ == '__main__':
    # 构造一个测试信号（例如随机信号）
    x = np.random.randn(10000)
    # 分析时刻，例如从 100 到 9900，每 100 个样本一个分析点
    naxis = np.arange(100, 9900, 100)
    # 设置中心频率和带宽（例如 fc = 0.1, fres = 0.02）
    fc = 0.1
    fres = 0.02
    f_est = ifdetect(x, naxis, fc, fres)
    print("Instantaneous frequency estimates:")
    print(f_est)