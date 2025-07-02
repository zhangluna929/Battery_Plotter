import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import stats
import os


def load_data(file_path, file_format='csv'):
    """
    本脚本支持CSV、Excel、JSON格式的原始数据
    :param file_path: 数据文件路径
    :param file_format: 数据文件格式 ('csv', 'excel', 'json')
    :return: 返回 pandas DataFrame 格式的数据
    """
    if file_format == 'csv':
        data = pd.read_csv(file_path)
    elif file_format == 'excel':
        data = pd.read_excel(file_path)
    elif file_format == 'json':
        data = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Please use 'csv', 'excel', or 'json'.")
    return data


def preprocess_data(data, noise_method='savgol', smooth_window=11, smooth_polyorder=3):
    """
    数据预处理，处理噪声和进行平滑。
    :param data: 原始数据
    :param noise_method: 噪声处理 ('savgol', 'moving_average', 'zscore')
    :param smooth_window: 平滑窗口大小
    :param smooth_polyorder: Savitzky-Golay平滑的多项式阶数
    :return: 预处理后的数据
    """
    if noise_method == 'savgol':
        # 使用 Savitzky-Golay 滤波器进行平滑
        smoothed_data = savgol_filter(data, window_length=smooth_window, polyorder=smooth_polyorder)
    elif noise_method == 'moving_average':
        # 使用简单的移动平均法平滑
        smoothed_data = data.rolling(window=smooth_window).mean()
    elif noise_method == 'zscore':
        # 使用 Z-score 标准化来去除异常值
        smoothed_data = stats.zscore(data)
    else:
        raise ValueError("Unsupported noise method. Please choose 'savgol', 'moving_average', or 'zscore'.")

    return smoothed_data


def plot_charge_discharge_curve(data, output_file='charge_discharge_curve.png'):
    """
    绘制充放电曲线。
    :param data: 实验数据 (含电压与时间数据)
    :param output_file: 输出图像
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


def plot_impedance_spectrum(data, output_file='impedance_spectrum.png'):
    """
    绘制阻抗谱
    :param data: 实验数据
    :param output_file: 输出图像
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


def plot_xrd_tem(data, output_file='xrd_tem.png'):
    """
    绘制XRD/TEM图。
    :param data: 实验数据
    :param output_file: 输出图像
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
    :param data: 实验数据 (包含循环次数和容量数据)
    :param output_file: 输出图像
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


def generate_reports(data, output_dir='output', file_format='csv', noise_method='savgol', smooth_window=11,
                     smooth_polyorder=3):
    """
    生成电池性能的报告图表。
    :param data: 实验数据
    :param output_dir: 输出目录
    :param file_format: 数据文件格式 ('csv', 'excel', 'json')
    :param noise_method: 噪声处理
    :param smooth_window: 平滑窗口大小
    :param smooth_polyorder: Savitzky-Golay平滑的多项式阶数
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    data = load_data(file_path=data, file_format=file_format)
    smoothed_voltage = preprocess_data(data['voltage'], noise_method=noise_method, smooth_window=smooth_window,
                                       smooth_polyorder=smooth_polyorder)
    smoothed_impedance = preprocess_data(data['impedance'], noise_method=noise_method, smooth_window=smooth_window,
                                         smooth_polyorder=smooth_polyorder)


    plot_charge_discharge_curve(data, output_file=os.path.join(output_dir, 'charge_discharge_curve.png'))


    plot_impedance_spectrum(data, output_file=os.path.join(output_dir, 'impedance_spectrum.png'))

    plot_xrd_tem(data, output_file=os.path.join(output_dir, 'xrd_tem.png'))

    plot_capacity_vs_cycle(data, output_file=os.path.join(output_dir, 'capacity_vs_cycle.png'))

    print("报告图表生成，保存至：", output_dir)


