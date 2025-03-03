import numpy as np
import os
import sys
import pyworld as pw
from estimate_ifsnr import estimate_ifsnr


def mag2db(x):
    """将幅值转换为 dB"""
    return 20 * np.log10(x)

def db2mag(x):
    """将 dB 转换为幅值"""
    return 10 ** (x / 20)


def make_random_signal(normalized_f0, n, at):
    """
    Python 版本的 make_random_signal
    ---------------------------------
    参数：
      normalized_f0 : 基频（f0/fs)）
      n  : 信号长度（采样点数）
      at : 噪声幅值（噪声缩放因子）
    
    返回：
      x     : 生成的随机信号（numpy 数组）
      naxis : 采样的索引（每 512 个点取一个样本）
    """
    # 生成 1 到 n 的采样点索引（与 MATLAB 的 (1:n)' 相同）
    xn = np.arange(1, n + 1)
    # 初始化信号
    x = np.zeros_like(xn, dtype=float)
    
    # 计算谐波个数
    nhar = int(np.floor(0.5 / normalized_f0))
    
    # 累加各谐波分量
    for i in range(1, nhar + 1):
        # 随机相位，均匀分布在 [0, 2*pi)
        p = np.random.rand() * 2 * np.pi
        # 随机幅值因子：10^(rand-0.5)
        h = 10 ** (np.random.rand() - 0.5)
        # 第一谐波固定为 1
        if i == 1:
            h = 1
        # 累加当前谐波分量
        x += h * np.sin(2 * np.pi * normalized_f0 * i * xn + p)
    
    # 添加高斯噪声，噪声幅值由 at 控制
    x += np.random.randn(*x.shape) * at
    
    # 生成 naxis，采样索引每 512 个点取一个
    naxis = np.arange(1, len(x) + 1, 512)
    
    return x, naxis


def make_random_dataset(num_datas, wtype = 'nuttall98', stype = 'hanning', fs = 8000, nch = 36):
    
    # 生成对数间隔的频率向量（归一化后）
    fc = np.logspace(np.log10(40), np.log10(1000), nch) / fs
    fres = fc.copy()

    data = []  # 用列表存放所有样本数据（每个元素为一个字典）
    num_samples = 0
    eps_val = 1e-10

    # 总共生成 num_datas 个样本
    for i in range(num_datas):
        # 打印进度
        print(f"{i+1}/{num_datas}")
        sys.stdout.flush()
        
        # global_snr = db2mag(rand * 100 - 50);
        global_snr = db2mag(np.random.rand() * 100 - 50)
        # f0 在 [40,1000] Hz 范围内随机，然后归一化
        f0_hz = (40 + (1000 - 40) * np.random.rand())
        f0 = f0_hz / fs
        
        # 生成随机信号，n=10000 采样点
        x, naxis = make_random_signal(f0, 10000, global_snr)
        
        # 估计特征，得到 SNR1, IF1, SNR2, IF2, SNR0
        SNR1, IF1, SNR2, IF2, SNR0 = estimate_ifsnr(x, naxis, fc, fres, wtype, stype)
        
        # 对 SNR 结果先进行下限截断，再转换为 dB
        SNR0_db = mag2db(np.maximum(SNR0, eps_val))
        SNR1_db = mag2db(np.maximum(SNR1, eps_val))
        SNR2_db = mag2db(np.maximum(SNR2, eps_val))
        # 瞬时频率乘以采样率恢复为 Hz
        IF1_scaled = IF1 * fs
        IF2_scaled = IF2 * fs

        # 保存本次样本的数据
        data.append({
            'SNR0': SNR0_db,
            'SNR1': SNR1_db,
            'SNR2': SNR2_db,
            'IF1': IF1_scaled,
            'IF2': IF2_scaled,
            'f0': f0_hz
        })
        
        # 累计样本数（每个样本的行数即分析帧数）
        num_samples += SNR0.shape[0]

    # 确保输出目录存在
    os.makedirs('data', exist_ok=True)
    
    # 针对每个频带单独生成样本文件
    for i in range(nch):
        # samples 矩阵大小为 (num_samples x 6)
        samples = np.zeros((num_samples, 6), dtype=np.float32)
        idx = 0
        # 遍历每个样本
        for j in range(len(data)):
            # 当前样本的帧数
            num = data[j]['SNR0'].shape[0]
            # 构造 F0 列向量，大小为 (num x 1)
            F0 = data[j]['f0'] * np.ones((num, 1), dtype=np.float32)
            # 依次取出对应频带 i 的各个特征列
            col1 = data[j]['SNR0'][:, i].reshape(-1, 1).astype(np.float32)
            col2 = data[j]['SNR1'][:, i].reshape(-1, 1).astype(np.float32)
            col3 = data[j]['SNR2'][:, i].reshape(-1, 1).astype(np.float32)
            col4 = data[j]['IF1'][:, i].reshape(-1, 1).astype(np.float32)
            col5 = data[j]['IF2'][:, i].reshape(-1, 1).astype(np.float32)
            # 拼接为 (num x 6) 的矩阵
            block = np.hstack([col1, col2, col3, col4, col5, F0])
            samples[idx:idx + num, :] = block
            idx += num

        # 文件名格式：data/samples-<频带索引>.f，注意 MATLAB 中频带索引从 1 开始
        filename = os.path.join('data', f'samples-{i+1}')
        
        # 以NumPy数组格式保存数据，这比直接写入二进制更符合Python风格
        # 使用allow_pickle=False确保更好的兼容性
        np.save(filename, samples, allow_pickle=False)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    
    # Test case 1: Generate a single random signal
    print("Test 1: Generating a single random signal")
    # Parameters
    f0_hz = 100
    fs = 8000
    f0 = f0_hz / fs  # 100 Hz normalized by 8000 Hz sampling rate
    signal_length = 5000
    noise_amplitude = 0.05
    
    # Generate signal
    start_time = time.time()
    x, naxis = make_random_signal(f0, signal_length, noise_amplitude)
    print(f"Signal generation time: {time.time() - start_time:.4f} seconds")
    print(f"Signal length: {len(x)}, Sample indices length: {len(naxis)}")
    
    # Plot signal
    try:
        plt.figure(figsize=(12, 8))
        plt.subplot(211)
        plt.plot(x[:1000])  # Plot first 1000 samples
        plt.title(f'Random Signal (f0={f0_hz:.1f} Hz, noise={noise_amplitude})')
        plt.xlabel('Sample index')
        plt.ylabel('Amplitude')
        
        # plot spectrum and f0 of the signal, calculate f0 use world dio
        # Convert signal to float64 and normalize for WORLD
        x_world = x.astype(np.float64)
        if np.max(np.abs(x_world)) > 1.0:
            x_world = x_world / np.max(np.abs(x_world))
        
        # Calculate F0 using WORLD's dio algorithm
        frame_period = 5.0  # Frame period in milliseconds
        f0_floor = 30.0  # Minimum F0
        f0_ceil = 1100.0  # Maximum F0
        
        # Extract F0 using dio (coarse estimation)
        f0, time_axis = pw.dio(x_world, fs, f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=frame_period)
        # Refine F0 using stonemask
        f0 = pw.stonemask(x_world, f0, time_axis, fs)
        
        # Plot F0 contour
        plt.subplot(212)
        plt.plot(time_axis, f0, 'r-')
        plt.title('F0 Contour')
        plt.xlabel('Time (s)')
        plt.ylabel('F0 (Hz)')
        plt.grid(True)
        
        # Set y-axis limits for F0 plot
        voiced_f0 = f0[f0 > 0]
        if len(voiced_f0) > 0:
            plt.ylim([0, np.max(voiced_f0) * 1.1])
            print(f"F0 range: {np.min(voiced_f0):.1f} - {np.max(voiced_f0):.1f} Hz")
        else:
            plt.ylim([0, 500])
            print("No voiced frames detected")
            
        plt.tight_layout()
        plt.savefig('signal_f0_analysis.png')
        plt.close()
        
        print("Signal and F0 analysis saved as 'signal_f0_analysis.png'")

    except Exception as e:
        print(f"Could not generate plot: {e}")
    
    # Test case 2: Small dataset generation (reduced parameters for quick testing)
    print("\nTest 2: Generating a small dataset")
    # Set smaller parameters for quick testing
    
    start_time = time.time()
    make_random_dataset(10)
    print(f"Dataset generation time: {time.time() - start_time:.4f} seconds")
    print("Dataset generated successfully!")

    print("\nTest cases completed!")