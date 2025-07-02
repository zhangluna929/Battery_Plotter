import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import stats
import os


def load_data(file_path, file_format='csv'):
    """
    载入XRD数据，支持 CSV、Excel、JSON 格式，灵活应对不同数据源。
    :param file_path: 数据文件的完整路径
    :param file_format: 数据文件格式 ('csv', 'excel', 'json')
    :return: 以 pandas DataFrame 格式返回数据
    """
    supported_formats = {
        'csv': pd.read_csv,
        'excel': pd.read_excel,
        'json': pd.read_json
    }
    
    if file_format not in supported_formats:
        raise ValueError(f"Unsupported format '{file_format}'. Please use 'csv', 'excel', or 'json'.")
    
    try:
        return supported_formats[file_format](file_path)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        raise


def smooth_data(data, method='savgol', window_size=11, poly_order=3):
    """
    对数据进行平滑处理，去除噪声，支持多种算法。
    :param data: 待平滑处理的数据
    :param method: 平滑方法（'savgol', 'moving_average', 'zscore'）
    :param window_size: 滤波窗口大小
    :param poly_order: Savitzky-Golay滤波器的多项式阶数
    :return: 平滑后的数据
    """
    if method == 'savgol':
        return savgol_filter(data, window_length=window_size, polyorder=poly_order)
    elif method == 'moving_average':
        return data.rolling(window=window_size).mean()
    elif method == 'zscore':
        return stats.zscore(data)
    else:
        raise ValueError(f"Method '{method}' is not supported. Choose 'savgol', 'moving_average', or 'zscore'.")


def find_peaks_in_data(data, threshold=0.5, min_distance=10):
    """
    自动检测数据中的峰值位置和峰值强度。
    :param data: 处理后的数据（通常是吸光度）
    :param threshold: 峰值高度阈值
    :param min_distance: 峰值间的最小距离
    :return: 峰值的位置和强度
    """
    peaks, properties = find_peaks(data, height=threshold, distance=min_distance)
    return peaks, properties['peak_heights']


def plot_voltage_curve(data, output_file='charge_discharge_curve.png'):
    """
    绘制充放电曲线：电压随时间变化的图像。
    :param data: 包含时间和电压的数据
    :param output_file: 保存图表的文件路径
    """
    plt.figure(figsize=(8, 6))
    plt.plot(data['time'], data['voltage'], label='Voltage', color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Charge/Discharge Curve')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_file)
    plt.close()


def plot_impedance(data, output_file='impedance_spectrum.png'):
    """
    绘制阻抗谱：频率与阻抗的关系图。
    :param data: 包含频率和阻抗的数据
    :param output_file: 保存图表的文件路径
    """
    plt.figure(figsize=(8, 6))
    plt.plot(data['frequency'], data['impedance'], label='Impedance', color='g')
    plt.xscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Impedance (Ohms)')
    plt.title('Impedance Spectrum')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_file)
    plt.close()


def plot_xrd(data, output_file='xrd_tem.png'):
    """
    绘制XRD或TEM图谱：展示2θ角度与强度的关系。
    :param data: 包含2theta和强度的数据
    :param output_file: 保存图表的文件路径
    """
    plt.figure(figsize=(8, 6))
    plt.plot(data['2theta'], data['intensity'], label='XRD/TEM', color='r')
    plt.xlabel('2θ (degrees)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('XRD/TEM Plot')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_file)
    plt.close()


def plot_capacity_vs_cycle(data, output_file='capacity_vs_cycle.png'):
    """
    绘制电池容量与循环次数的关系图。
    :param data: 包含循环次数和容量的数据
    :param output_file: 保存图表的文件路径
    """
    plt.figure(figsize=(8, 6))
    plt.plot(data['cycle'], data['capacity'], label='Capacity', color='m')
    plt.xlabel('Cycle Number')
    plt.ylabel('Capacity (Ah)')
    plt.title('Capacity vs Cycle')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_file)
    plt.close()


def generate_report(data, output_dir='output', file_format='csv', noise_method='savgol', smooth_window=11,
                    smooth_polyorder=3):
    """
    根据实验数据生成电池性能的报告，包括图表。
    :param data: 输入数据文件路径
    :param output_dir: 图表保存的目录
    :param file_format: 数据文件格式 ('csv', 'excel', 'json')
    :param noise_method: 噪声处理方法
    :param smooth_window: 平滑窗口大小
    :param smooth_polyorder: Savitzky-Golay平滑的多项式阶数
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载并预处理数据
    data = load_data(file_path=data, file_format=file_format)
    smoothed_voltage = smooth_data(data['voltage'], method=noise_method, window_size=smooth_window,
                                   poly_order=smooth_polyorder)
    smoothed_impedance = smooth_data(data['impedance'], method=noise_method, window_size=smooth_window,
                                     poly_order=smooth_polyorder)

    # 绘制各类图表
    plot_voltage_curve(data, output_file=os.path.join(output_dir, 'charge_discharge_curve.png'))
    plot_impedance(data, output_file=os.path.join(output_dir, 'impedance_spectrum.png'))
    plot_xrd(data, output_file=os.path.join(output_dir, 'xrd_tem.png'))
    plot_capacity_vs_cycle(data, output_file=os.path.join(output_dir, 'capacity_vs_cycle.png'))

    print(f"报告图表已生成并保存在：{output_dir}")
