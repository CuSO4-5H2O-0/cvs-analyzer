#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVS 整平剂浓度自动计算脚本
=========================
基于循环伏安溶出法（CVS）标准加入法，计算待测液中整平剂浓度。

原理：
1. 对 CV 最后一圈积分，得到沉积（充电）和溶出（放电）峰面积
2. 响应值 R = 空白面积 - 测量面积，与有效浓度线性相关
3. 标准加入法线性拟合，x 截距得待测液浓度

积分窗口：
- 溶出峰（放电）：左端=电流0电位 → 右端=0.5V
- 沉积区（充电）：负扫开始明显下降 → 正扫至电流0结束（取溶出峰后过零点）

泵定义：
- 泵1: VMS（活化用）
- 泵2: 测试底液
- 泵3: 待测液（未知浓度，体积 V_TEST）
- 泵4: 标准添加液L（整平剂，已知浓度 C_STD）
- 泵5: 纯A（促进剂）
- 泵6: 纯S（抑制剂）

使用方法：
  python analyze_cvs.py [数据文件夹路径]

输出：
  - 终端打印结果表和浓度
  - MALT_plot.png 标准加入法线性图

响应类型（通过 RESPONSE_TYPE 配置）：
  - 'stripping': 使用溶出峰（放电）面积差（默认）
  - 'charging': 使用沉积区（充电）面积差（与原始 Excel 一致）
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 全局字体设置：Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学符号与 Times New Roman 匹配
plt.rcParams['font.size'] = 13
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
import os
import re
from scipy import stats

# ============================================================
# 配置参数
# ============================================================
C_STD = 50.0          # 标准液整平剂浓度
V_TEST = 5000.0        # 待测液体积 (uL)
V_BASE = 5000.0        # 底液体积 (uL)
SCAN_RATE = 0.2        # 扫速 (V/s)
STEP_V = 0.002         # 电位步长 (V)
DISCHARGE_V_MAX = 0.5  # 溶出峰右端最大电位 (V)
RESPONSE_TYPE = 'stripping'  # 响应类型: 'stripping'(溶出峰) 或 'charging'(沉积峰)
N_STD_POINTS = 2       # 使用的标准添加点数，0=使用所有点

# ============================================================
# 工具函数
# ============================================================

def parse_filename(filename):
    """
    解析 CVS 文件名，提取泵参数。
    文件名格式: CV_{P1}-{P2}-{P3}-{P4}-{P5}-{P6}_{扫速}_{转速}_{时间}.txt
    """
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) < 3:
        return None
    pump_str = parts[1]
    pump_vals = [int(x) for x in pump_str.split('-')]
    return {
        'pumps': pump_vals,
        'file': basename
    }


def read_cv_data(filepath):
    """
    读取 CVS txt 文件，返回 (potential, current) 数组。
    跳过前5行表头和末尾的 0.0 0.0 行。
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines[5:-1]:  # 跳过表头(5行)和末尾空行
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                data.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    return np.array(data)


def detect_cycles(data):
    """
    检测 CV 循环转折点，返回转折点索引列表。
    """
    pot = data[:, 0]
    diffs = np.diff(pot)
    direction = np.zeros(len(diffs), dtype=int)
    direction[diffs > 0.0001] = 1      # 电位上升
    direction[diffs < -0.0001] = -1    # 电位下降

    reversals = []
    cur_dir = direction[0]
    for i in range(1, len(direction)):
        if direction[i] != 0 and direction[i] != cur_dir:
            if cur_dir != 0:
                reversals.append(i)
            cur_dir = direction[i]
    return reversals


def find_zero_crossing(data, start_idx, direction='pos_to_neg'):
    """
    找到电流过零点的插值电位。
    direction='pos_to_neg': 正→负
    direction='neg_to_pos': 负→正
    """
    for i in range(start_idx, len(data) - 1):
        if direction == 'pos_to_neg':
            if data[i, 1] >= 0 and data[i+1, 1] < 0:
                frac = data[i, 1] / (data[i, 1] - data[i+1, 1])
                return data[i, 0] + frac * (data[i+1, 0] - data[i, 0])
        elif direction == 'neg_to_pos':
            if data[i, 1] <= 0 and data[i+1, 1] > 0:
                frac = -data[i, 1] / (data[i+1, 1] - data[i, 1])
                return data[i, 0] + frac * (data[i+1, 0] - data[i, 0])
    return None


def find_deposition_onset(fwd_sweep):
    """
    在正向扫（负方向）中找到沉积开始的电位。
    使用两个标准：
    1. 电流首次转负
    2. 电流低于初始平台期的平均值减去3倍噪声
    取电位更负的那个（更保守的估计）。
    """
    if len(fwd_sweep) < 20:
        return fwd_sweep[0, 0]

    # 用前10点估计平台期电流
    plateau = np.mean(fwd_sweep[:10, 1])
    noise = np.std(fwd_sweep[:10, 1])

    # 条件1：电流首次转负
    v_first_neg = None
    for i in range(len(fwd_sweep)):
        if fwd_sweep[i, 1] < 0:
            v_first_neg = fwd_sweep[i, 0]
            break

    # 条件2：电流明显低于平台期
    v_below_threshold = None
    threshold = plateau - 3 * noise
    for i in range(len(fwd_sweep)):
        if fwd_sweep[i, 1] < threshold:
            v_below_threshold = fwd_sweep[i, 0]
            break

    # 取两者中电位更负的（更保守）
    candidates = [v for v in [v_first_neg, v_below_threshold] if v is not None]
    if not candidates:
        return fwd_sweep[0, 0]
    return min(candidates)


def integrate_stripping(bwd_sweep):
    """
    溶出峰积分（放电）：
    左端=电流0电位（负→正），右端=min(0.5V, sweep_end)
    积分 ∫I·dV
    """
    v_start = find_zero_crossing(bwd_sweep, 0, 'neg_to_pos')
    if v_start is None:
        v_start = bwd_sweep[0, 0]

    v_end = min(DISCHARGE_V_MAX, bwd_sweep[-1, 0])

    # 在数据中找对应区间
    mask = (bwd_sweep[:, 0] >= v_start) & (bwd_sweep[:, 0] <= v_end)
    segment = bwd_sweep[mask]

    if len(segment) < 2:
        return 0.0, v_start, v_end

    area = np.trapz(segment[:, 1], segment[:, 0])
    return area, v_start, v_end


def integrate_deposition(fwd_sweep, bwd_sweep):
    """
    沉积积分（充电）：
    左端=负扫开始明显下降处，右端=正扫至电流0处（取溶出峰之后的过零点）。
    积分覆盖从沉积开始 → 经过转折 → 到电流归零的完整路径。
    积分 ∫I·dV
    """
    # 左端：沉积开始电位
    v_start = find_deposition_onset(fwd_sweep)

    # 右端：在反向扫中，溶出峰过后电流归零处（正→负）
    # 需跳过溶出峰之前的噪声过零点，从溶出峰之后的区域搜索
    # 用 0.3V 作为搜索起点（溶出峰在 ~0.15V 左右结束）
    start_idx = np.searchsorted(bwd_sweep[:, 0], 0.3)
    v_end = find_zero_crossing(bwd_sweep, start_idx, 'pos_to_neg')
    if v_end is None:
        v_end = bwd_sweep[-1, 0]

    # 构建完整路径：从沉积开始→转折点→电流归零
    # 前向扫部分：V从v_start到转折点
    fwd_segment = fwd_sweep[fwd_sweep[:, 0] >= v_start]
    if len(fwd_segment) < 2:
        return 0.0, v_start, v_end

    # 反向扫部分：从转折点到v_end
    bwd_segment = bwd_sweep[bwd_sweep[:, 0] <= v_end]
    if len(bwd_segment) < 2:
        bwd_segment = bwd_sweep

    # 合并路径（顺序：前向扫后接反向扫）
    full_path = np.vstack([fwd_segment, bwd_segment])

    area = np.trapz(full_path[:, 1], full_path[:, 0])
    return area, v_start, v_end


def group_by_block(files_info):
    """
    按时间序列分组为 block。
    每个 block 包含：1个VMS + 1个空白 + 4个标准添加（0,100,200,300 uL）
    """
    # 按时间排序
    files_info.sort(key=lambda x: x['time'])

    blocks = []
    current_block = []
    for f in files_info:
        p = f['pumps']
        if p[0] == 6000:  # VMS标记序列开始
            if current_block:
                blocks.append(current_block)
            current_block = [f]
        elif p[1] == 5000:  # 测试序列
            current_block.append(f)
    if current_block:
        blocks.append(current_block)
    return blocks


def extract_time_from_file(filename):
    """从文件名提取时间"""
    # 格式: ..._2026-05-14-10-38-21.txt
    match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', filename)
    if match:
        return match.group(1)
    return ''


# ============================================================
# 主分析函数
# ============================================================

def analyze_folder(folder_path):
    """
    分析文件夹内所有 CVS txt 文件。
    """
    # 1. 读取所有文件
    all_files = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith('.txt'):
            continue
        filepath = os.path.join(folder_path, fname)
        info = parse_filename(fname)
        if info is None:
            continue
        info['filepath'] = filepath
        info['time'] = extract_time_from_file(fname)

        # 读取数据
        data = read_cv_data(filepath)
        if len(data) < 100:
            continue
        info['data'] = data
        all_files.append(info)

    print(f"共读取 {len(all_files)} 个 CVS 文件")

    # 2. 检测循环并积分
    results = []
    for f in all_files:
        data = f['data']
        reversals = detect_cycles(data)
        if len(reversals) < 3:
            print(f"  跳过 {f['file']}: 循环检测失败")
            continue

        # 取最后一圈
        last_fwd_start = reversals[-2]
        last_bwd_end = len(data)
        last_fwd = data[last_fwd_start:reversals[-1]]
        last_bwd = data[reversals[-1]:last_bwd_end]

        # 积分
        discharge_area, ds_v_start, ds_v_end = integrate_stripping(last_bwd)
        charge_area, ch_v_start, ch_v_end = integrate_deposition(last_fwd, last_bwd)

        f['charge_area'] = charge_area
        f['discharge_area'] = discharge_area
        f['ch_v_start'] = ch_v_start
        f['ch_v_end'] = ch_v_end
        f['ds_v_start'] = ds_v_start
        f['ds_v_end'] = ds_v_end

    # 3. 按 block 分组并计算
    blocks = group_by_block(all_files)

    block_results = []
    for bi, block in enumerate(blocks):
        print(f"\n===== Block {bi+1} =====")

        # 在 block 中找空白（P3=0, P4=0）
        baselines = [f for f in block if f['pumps'][2] == 0 and f['pumps'][3] == 0 and f['pumps'][0] != 6000]
        if not baselines:
            print(f"  Block {bi+1}: 未找到空白")
            continue
        # 用最后一个空白（最接近样品测量的）
        baseline = baselines[-1]

        # 找样品 + 标准添加（P3=5000）
        samples = [f for f in block if f['pumps'][2] == 5000]
        # 按 P4（标准添加量）排序
        samples.sort(key=lambda x: x['pumps'][3])

        print(f"  空白: {baseline['file']}")
        print(f"  积分窗口 - 沉积: {baseline.get('ch_v_start',0):.4f}V → {baseline.get('ch_v_end',0):.4f}V")
        print(f"           溶出: {baseline.get('ds_v_start',0):.4f}V → {baseline.get('ds_v_end',0):.4f}V")

        # 计算响应值 R = A_blank - A_sample
        if RESPONSE_TYPE == 'charging':
            A0 = baseline['charge_area']
            area_label = '沉积'
            print(f"  空白沉积峰面积: {A0:.5f}")
        else:
            A0 = baseline['discharge_area']
            area_label = '溶出'
            print(f"  空白溶出峰面积: {A0:.5f}")

        std_points = []  # (V_std, C_eff, R)
        for s in samples:
            if RESPONSE_TYPE == 'charging':
                A = s['charge_area']
            else:
                A = s['discharge_area']
            R = A0 - A  # 响应值（正值表示面积减少）

            V_std = s['pumps'][3]  # 标准添加体积 (uL)
            # 有效浓度 = V_std × C_STD / V_total
            V_total = V_BASE + V_TEST + V_std
            C_eff = V_std * C_STD / V_total if V_std > 0 else 0.0

            std_points.append({
                'file': s['file'],
                'V_std': V_std,
                'C_eff': C_eff,
                'discharge_area': A,
                'R': R,
                'charge_area': s.get('charge_area', 0)
            })
            print(f"  添加 {V_std:3d} uL: {area_label}面积={A:.5f}, R={R:.5f}, C_eff={C_eff:.4f}")

        # 根据 N_STD_POINTS 选择参与拟合的点数
        # std_points 已按添加量升序排列 [0uL, 100uL, 200uL, 300uL]
        # N_STD_POINTS 表示使用的标准添加点数（不含0点）
        if N_STD_POINTS > 0 and len(std_points) > N_STD_POINTS:
            fit_points = std_points[:N_STD_POINTS + 1]  # +1 包含0点
            print(f"  选择前 {N_STD_POINTS} 个添加点进行拟合（共 {len(std_points)-1} 个）")
        else:
            fit_points = std_points

        # 标准加入法线性拟合: R vs C_eff
        if len(fit_points) >= 2:
            C = np.array([p['C_eff'] for p in fit_points])
            R = np.array([p['R'] for p in fit_points])

            # 线性回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(C, R)

            # x 截距（R=0 时的 C_eff，为负值）
            C_sample_eff = -intercept / slope if slope != 0 else 0

            # 换算回待测液原始浓度（取绝对值）
            # |C_sample_eff| = C_x × V_TEST / (V_BASE + V_TEST)
            C_x = abs(C_sample_eff) * (V_BASE + V_TEST) / V_TEST

            block_results.append({
                'block': bi + 1,
                'points': fit_points,
                'A0': A0,
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'C_sample_eff': C_sample_eff,
                'C_x': C_x,
            })

            print(f"\n  >>> 线性拟合:")
            print(f"      R = {slope:.5f} × C_eff + ({intercept:.5f})")
            print(f"      R^2 = {r_value**2:.4f}")
            print(f"      x截距 (C_sample_eff) = {C_sample_eff:.4f}")
            print(f"      待测液整平剂浓度 = {C_x:.4f}")

    # 聚合分析：合并所有 block 的数据点，统一拟合
    if len(block_results) >= 2:
        all_C = []
        all_R = []
        for r in block_results:
            for p in r['points']:
                all_C.append(p['C_eff'])
                all_R.append(p['R'])
        if len(all_C) >= 3:
            slope_p, intercept_p, r_p, _, _ = stats.linregress(all_C, all_R)
            C_sample_eff_p = -intercept_p / slope_p
            C_x_p = abs(C_sample_eff_p) * (V_BASE + V_TEST) / V_TEST
            pooled_result = {
                'block': 'All',
                'points': [{'C_eff': c, 'R': r, 'V_std': 0, 'discharge_area': 0, 'charge_area': 0, 'file': 'pooled'} for c, r in zip(all_C, all_R)],
                'A0': 0,
                'slope': slope_p,
                'intercept': intercept_p,
                'r_squared': r_p ** 2,
                'C_sample_eff': C_sample_eff_p,
                'C_x': C_x_p,
            }
            block_results.append(pooled_result)
            print(f"\n  >>> 聚合拟合 (All blocks pooled):")
            print(f"      R = {slope_p:.5f} × C_eff + ({intercept_p:.5f})")
            print(f"      R^2 = {r_p**2:.4f}")
            print(f"      C_x (待测液整平剂浓度) = {C_x_p:.4f}")

    return block_results, all_files


# ============================================================
# CSV 导出
# ============================================================

def export_csv(results, output_path):
    """
    导出浓度计算结果表为 CSV 文件。
    包含：各点峰面积详情 + 各 block 拟合结果 + 聚合结果。
    """
    import csv

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)

        # 表头信息
        writer.writerow(['CVS 整平剂浓度计算结果'])
        writer.writerow([f'C_STD={C_STD}, V_TEST={V_TEST}, V_BASE={V_BASE}, 响应类型={RESPONSE_TYPE}'])
        writer.writerow([])

        # === 各测量点详情 ===
        area_label = '溶出面积' if RESPONSE_TYPE == 'stripping' else '沉积面积'
        writer.writerow(['各测量点峰面积详情'])
        writer.writerow(['Block', 'P4(uL)', 'C_eff', area_label, 'R值', '文件名'])

        individual_results = [r for r in results if r['block'] != 'All']
        for r in individual_results:
            for p in r['points']:
                area_val = p['discharge_area'] if RESPONSE_TYPE == 'stripping' else p['charge_area']
                writer.writerow([
                    f'Block {r["block"]}',
                    p['V_std'],
                    f'{p["C_eff"]:.4f}',
                    f'{area_val:.5f}',
                    f'{p["R"]:.5f}',
                    os.path.basename(p['file'])
                ])

        writer.writerow([])

        # === 拟合结果汇总 ===
        writer.writerow(['拟合结果汇总'])
        writer.writerow(['Block', 'R²', '斜率', '截距', 'C_sample_eff', 'C_x (浓度)'])

        for r in results:
            label = f'Block {r["block"]}' if r['block'] != 'All' else 'Pooled'
            writer.writerow([
                label,
                f'{r["r_squared"]:.4f}',
                f'{r["slope"]:.5f}',
                f'{r["intercept"]:.5f}',
                f'{r["C_sample_eff"]:.4f}',
                f'{r["C_x"]:.4f}',
            ])

        # 统计
        C_x_values = [r['C_x'] for r in results if r['block'] != 'All']
        if len(C_x_values) > 1:
            mean_cx = np.mean(C_x_values)
            std_cx = np.std(C_x_values, ddof=1)
            rsd = std_cx / mean_cx * 100 if mean_cx != 0 else 0
            writer.writerow(['Mean', '', '', '', '', f'{mean_cx:.4f}'])
            writer.writerow(['SD', '', '', '', '', f'{std_cx:.4f}'])
            writer.writerow(['RSD%', '', '', '', '', f'{rsd:.2f}'])

    print(f"结果表已保存: {output_path}")


# ============================================================
# 绘图
# ============================================================

def plot_malt(results, output_path=None):
    """
    绘制 MALT 图（标准加入法线性图）。
    每个 block 一条线，显示 R vs C_eff 的拟合。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左图：各 block 的标曲
    ax1 = axes[0]
    individual_results = [r for r in results if r['block'] != 'All']
    pooled_results = [r for r in results if r['block'] == 'All']
    colors = plt.cm.tab10(np.linspace(0, 1, len(individual_results)))

    for i, r in enumerate(individual_results):
        C = np.array([p['C_eff'] for p in r['points']])
        R_data = np.array([p['R'] for p in r['points']])

        ax1.scatter(C, R_data, color=colors[i], s=60, zorder=5, label=f'Block {r["block"]}')

        # 拟合线
        C_fit = np.linspace(min(C.min(), r['C_sample_eff'] - 0.5),
                            max(C.max(), 0), 100)
        R_fit = r['slope'] * C_fit + r['intercept']
        ax1.plot(C_fit, R_fit, '--', color=colors[i], alpha=0.5, linewidth=1.2)

        # 标注 x 截距
        ax1.axvline(x=r['C_sample_eff'], color=colors[i], linestyle=':',
                    alpha=0.3, linewidth=0.8)

        # 在数据旁标注 R²
        ax1.annotate(f'R²={r["r_squared"]:.3f}',
                     xy=(C.max(), R_data[-1]),
                     xytext=(C.max() * 0.6, R_data[-1] * 0.5 + R_data[0] * 0.5),
                     fontsize=11, color=colors[i],
                     alpha=0.8, ha='right')

    # 聚合拟合线（加粗黑线）
    for r in pooled_results:
        C_all = np.array([p['C_eff'] for p in r['points']])
        ax1.scatter(C_all, [p['R'] for p in r['points']],
                   color='black', s=15, zorder=4, alpha=0.4, label='_nolegend_')
        C_fit = np.linspace(min(C_all.min(), r['C_sample_eff'] - 0.5),
                            max(C_all.max(), 0), 100)
        R_fit = r['slope'] * C_fit + r['intercept']
        ax1.plot(C_fit, R_fit, '-', color='black', linewidth=2.5, alpha=0.8,
                label=f"Pooled (C$_x$={r['C_x']:.2f})")
        ax1.axvline(x=r['C_sample_eff'], color='black', linestyle='--',
                    alpha=0.6, linewidth=1)
        ax1.annotate(f'C$_x$={r["C_x"]:.2f}',
                     xy=(r['C_sample_eff'], 0),
                     xytext=(r['C_sample_eff'] - 0.25, max(C_all)*0.05),
                     fontsize=12, color='black',
                     arrowprops=dict(arrowstyle='->', color='black'))
        # 标注 R²
        ax1.text(0.95, 0.05, f'Pooled R²={r["r_squared"]:.3f}',
                 transform=ax1.transAxes, fontsize=12,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Effective concentration C_eff (std addition)')
    ax1.set_ylabel('Response R = A0 - A')
    ax1.set_title('Standard Addition Calibration Curve')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 右图：各添加点的溶出峰面积
    ax2 = axes[1]
    for i, r in enumerate(individual_results):
        V = np.array([p['V_std'] for p in r['points']])
        A = np.array([p['discharge_area'] for p in r['points']])

        ax2.scatter(V, A, color=colors[i], s=60, zorder=5, label=f'Block {r["block"]}')
        ax2.plot(V, A, '-o', color=colors[i], alpha=0.5, markersize=5)

    if individual_results:
        ax2.axhline(y=individual_results[0].get('A0', 0), color='gray', linestyle='--', alpha=0.5, label='Blank A0')
    ax2.set_xlabel('Standard addition volume (uL)')
    ax2.set_ylabel('Stripping peak area')
    ax2.set_title('Stripping Area vs Addition')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"\nMALT plot saved: {output_path}")


def plot_individual_malt(result, output_path=None):
    """
    绘制单个 block 的 MALT 图（标准加入法线性图）。
    左图：R vs C_eff 拟合；右图：峰面积 vs 添加体积。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左图：标曲
    ax1 = axes[0]
    C = np.array([p['C_eff'] for p in result['points']])
    R_data = np.array([p['R'] for p in result['points']])

    ax1.scatter(C, R_data, color='#2196F3', s=60, zorder=5, label='Data points')

    # 拟合线
    C_fit = np.linspace(min(C.min(), result['C_sample_eff'] - 0.5),
                        max(C.max(), 0), 100)
    R_fit = result['slope'] * C_fit + result['intercept']
    ax1.plot(C_fit, R_fit, '-', color='#FF5722', linewidth=2, alpha=0.8,
             label=f"Fit: R={result['slope']:.4f}C+({result['intercept']:.4f})")

    # x 截距
    ax1.axvline(x=result['C_sample_eff'], color='#FF5722', linestyle='--',
                alpha=0.5, linewidth=1)
    ax1.annotate(f'C$_x$={result["C_x"]:.2f}',
                 xy=(result['C_sample_eff'], 0),
                 xytext=(result['C_sample_eff'] - 0.35, max(R_data) * 0.7),
                 fontsize=13, color='#FF5722',
                 arrowprops=dict(arrowstyle='->', color='#FF5722'))

    # R² 标注（右下角，避免遮盖数据点）
    ax1.text(0.95, 0.05, f'R² = {result["r_squared"]:.4f}',
             transform=ax1.transAxes, fontsize=14,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    # 空白基线
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Effective concentration C_eff')
    ax1.set_ylabel('Response R = A0 - A')
    ax1.set_title(f'Block {result["block"]} — Standard Addition')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 右图：峰面积 vs 添加体积
    ax2 = axes[1]
    V = np.array([p['V_std'] for p in result['points']])
    A = np.array([p['discharge_area'] for p in result['points']])

    ax2.scatter(V, A, color='#4CAF50', s=70, zorder=5, label='Stripping area')
    ax2.plot(V, A, '-o', color='#4CAF50', alpha=0.6, markersize=7)
    ax2.axhline(y=result.get('A0', 0), color='gray', linestyle='--',
                alpha=0.5, label=f"Blank A0={result.get('A0',0):.4f}")

    ax2.set_xlabel('Standard addition volume (uL)')
    ax2.set_ylabel('Stripping peak area')
    ax2.set_title(f'Block {result["block"]} — Area vs Addition')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Individual MALT plot saved: {output_path}")


# ============================================================
# 结果输出
# ============================================================

def print_results_table(results):
    """打印各 block 的详细结果表"""
    print("\n" + "="*70)
    print("待测液整平剂浓度计算结果")
    print("="*70)

    print(f"\n{'Block':<8} {'R²':<10} {'斜率':<12} {'截距':<12} {'C_sample_eff':<15} {'C_x (浓度)':<15}")
    print("-"*70)

    C_x_values = []
    for r in results:
        label = f"Block {r['block']}" if r['block'] != 'All' else 'Pooled'
        print(f"{label:<8} {r['r_squared']:<10.4f} {r['slope']:<12.5f} "
              f"{r['intercept']:<12.5f} {r['C_sample_eff']:<15.4f} {r['C_x']:<15.4f}")
        if r['block'] != 'All':
            C_x_values.append(r['C_x'])

    if len(C_x_values) > 1:
        mean_cx = np.mean(C_x_values)
        std_cx = np.std(C_x_values, ddof=1)
        rsd = std_cx / mean_cx * 100 if mean_cx != 0 else 0
        print("-"*70)
        print(f"{'Mean':<8} {'':<10} {'':<12} {'':<12} {'':<15} {mean_cx:<15.4f}")
        print(f"{'SD':<8} {'':<10} {'':<12} {'':<12} {'':<15} {std_cx:<15.4f}")
        print(f"{'RSD%':<8} {'':<10} {'':<12} {'':<12} {'':<15} {rsd:<15.2f}")

    # 打印各点峰面积
    print("\n" + "="*90)
    print("各测量点峰面积详情")
    print("="*90)
    print(f"{'Block':<8} {'P4(uL)':<10} {'C_eff':<10} {'溶出面积':<12} {'沉积面积':<12} {'R值':<12} {'文件':<30}")
    print("-"*90)
    for r in results:
        if r['block'] == 'All':
            continue
        for p in r['points']:
            print(f"{r['block']:<8} {p['V_std']:<10} {p['C_eff']:<10.4f} "
                  f"{p['discharge_area']:<12.5f} {p['charge_area']:<12.5f} "
                  f"{p['R']:<12.5f} {os.path.basename(p['file'])[:28]:<28}")


def plot_integration_window(data, reversals, title="CV 曲线与积分窗口"):
    """可视化积分窗口"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data[:, 0], data[:, 1], 'b-', linewidth=0.8, label='Full CV')

    # 标注最后一圈
    if len(reversals) >= 3:
        last_start = reversals[-2]
        last_end = len(data)
        ax.plot(data[last_start:reversals[-1], 0], data[last_start:reversals[-1], 1],
                'g-', linewidth=1.5, alpha=0.7, label='Last forward (deposition)')
        ax.plot(data[reversals[-1]:last_end, 0], data[reversals[-1]:last_end, 1],
                'r-', linewidth=1.5, alpha=0.7, label='Last backward (stripping)')

    ax.set_xlabel('Potential (V)')
    ax.set_ylabel('Current (mA)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    import sys
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # 默认路径
    default_path = r'D:\文档\科研\实验数据\2026\5\14-box-12-23-STD-0.2\txt_box'

    # 在 Jupyter/IPython 环境中 sys.argv 可能包含内核参数，过滤掉
    import glob
    folder = default_path
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            folder = arg
            break

    if not os.path.isdir(folder):
        print(f"错误: 文件夹不存在 - {folder}")
        sys.exit(1)

    print(f"分析文件夹: {folder}")
    print(f"参数: C_STD={C_STD}, V_TEST={V_TEST}, V_BASE={V_BASE}")

    results, all_files = analyze_folder(folder)

    if not results:
        print("未生成有效结果，请检查数据格式")
        sys.exit(1)

    # 输出结果表
    print_results_table(results)

    # 导出 CSV 结果表
    csv_path = os.path.join(os.path.dirname(folder), '计算结果.csv')
    export_csv(results, csv_path)

    # 绘制 MALT 图
    plot_path = os.path.join(os.path.dirname(folder), 'MALT_plot.png')
    plot_malt(results, plot_path)

    # 绘制每个 block 的单独 MALT 图
    parent_dir = os.path.dirname(folder)
    for r in results:
        if r['block'] == 'All':
            continue
        individual_path = os.path.join(parent_dir, f'MALT_plot_Block{r["block"]}.png')
        plot_individual_malt(r, individual_path)

    print("\n分析完成!")
