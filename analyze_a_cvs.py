#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVS 促进剂(A)浓度自动计算脚本 — MLAT 法
=========================================
基于循环伏安溶出法（CVS）标准加入法，计算待测液中促进剂浓度。

泵配置（与测L不同）：
- 泵1: VMS（仅用于活化电极，不参与测试）
- 泵2: 测量底液（体积 V_BASE）
- 泵3: 空
- 泵4: 待测液(STD) + 标准添加液（同一泵，需区分首次添加和后续加标）
- 泵5: 空
- 泵6: 空

测量流程：
VMS活化 → 扫底液（泵2）→ 加待测液（泵4，首次）→ 多次加标（泵4，后续）

由于待测液和加标均使用 STD 且通过同一泵添加，泵4 的第一次非零体积为
待测液添加，后续体积为累积加标。

使用方法：
  python analyze_a_cvs.py [数据文件夹路径]

输出：
  - 终端打印结果表和浓度
  - MLAT_plot.png / MLAT_plot_BlockN.png 标准加入法线性图
  - 计算结果.csv
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
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
C_STD = 23.0            # 标准液(STD)中 A 的浓度（也是待测液的理论浓度）
SCAN_RATE = 0.2         # 测试扫速 (V/s)，0.1V/s 的 VMS 文件会被排除
STEP_V = 0.002          # 电位步长 (V)
DISCHARGE_V_MAX = 0.5   # 溶出峰右端最大电位 (V)
RESPONSE_TYPE = 'stripping'  # 响应类型: 'stripping'(溶出峰) 或 'charging'(沉积峰)
N_STD_POINTS = 0        # 使用的标准添加点数（不含样品点），0=使用所有点

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
    """读取 CVS txt 文件，跳过前5行表头和末尾空行。"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines[5:-1]:
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                data.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    return np.array(data)


def detect_cycles(data):
    """检测 CV 循环转折点，返回转折点索引列表。"""
    pot = data[:, 0]
    diffs = np.diff(pot)
    direction = np.zeros(len(diffs), dtype=int)
    direction[diffs > 0.0001] = 1
    direction[diffs < -0.0001] = -1

    reversals = []
    cur_dir = direction[0]
    for i in range(1, len(direction)):
        if direction[i] != 0 and direction[i] != cur_dir:
            if cur_dir != 0:
                reversals.append(i)
            cur_dir = direction[i]
    return reversals


def find_zero_crossing(data, start_idx, direction='pos_to_neg'):
    """找到电流过零点的插值电位。"""
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
    """在正向扫（负方向）中找到沉积开始的电位。"""
    if len(fwd_sweep) < 20:
        return fwd_sweep[0, 0]

    plateau = np.mean(fwd_sweep[:10, 1])
    noise = np.std(fwd_sweep[:10, 1])

    v_first_neg = None
    for i in range(len(fwd_sweep)):
        if fwd_sweep[i, 1] < 0:
            v_first_neg = fwd_sweep[i, 0]
            break

    v_below_threshold = None
    threshold = plateau - 3 * noise
    for i in range(len(fwd_sweep)):
        if fwd_sweep[i, 1] < threshold:
            v_below_threshold = fwd_sweep[i, 0]
            break

    candidates = [v for v in [v_first_neg, v_below_threshold] if v is not None]
    if not candidates:
        return fwd_sweep[0, 0]
    return min(candidates)


def integrate_stripping(bwd_sweep):
    """溶出峰积分（放电）：左端=电流0电位（负→正），右端=min(0.5V, sweep_end)。"""
    v_start = find_zero_crossing(bwd_sweep, 0, 'neg_to_pos')
    if v_start is None:
        v_start = bwd_sweep[0, 0]

    v_end = min(DISCHARGE_V_MAX, bwd_sweep[-1, 0])

    mask = (bwd_sweep[:, 0] >= v_start) & (bwd_sweep[:, 0] <= v_end)
    segment = bwd_sweep[mask]

    if len(segment) < 2:
        return 0.0, v_start, v_end

    area = np.trapz(segment[:, 1], segment[:, 0])
    return area, v_start, v_end


def integrate_deposition(fwd_sweep, bwd_sweep):
    """沉积积分（充电）。"""
    v_start = find_deposition_onset(fwd_sweep)

    start_idx = np.searchsorted(bwd_sweep[:, 0], 0.3)
    v_end = find_zero_crossing(bwd_sweep, start_idx, 'pos_to_neg')
    if v_end is None:
        v_end = bwd_sweep[-1, 0]

    fwd_segment = fwd_sweep[fwd_sweep[:, 0] >= v_start]
    if len(fwd_segment) < 2:
        return 0.0, v_start, v_end

    bwd_segment = bwd_sweep[bwd_sweep[:, 0] <= v_end]
    if len(bwd_segment) < 2:
        bwd_segment = bwd_sweep

    full_path = np.vstack([fwd_segment, bwd_segment])
    area = np.trapz(full_path[:, 1], full_path[:, 0])
    return area, v_start, v_end


def group_by_block(files_info):
    """
    按时间序列分组为 block。
    VMS（泵1=6000）标记 block 开始。
    每个 block 包含：1个VMS + 1个空白 + 1个样品(首次泵4) + N个标准添加(后续泵4)
    """
    files_info.sort(key=lambda x: x['time'])

    blocks = []
    current_block = []
    for f in files_info:
        p = f['pumps']
        if p[0] == 6000:  # VMS 标记 block 开始
            if current_block:
                blocks.append(current_block)
            current_block = [f]
        else:
            current_block.append(f)
    if current_block:
        blocks.append(current_block)
    return blocks


def extract_time_from_file(filename):
    """从文件名提取时间。"""
    match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', filename)
    if match:
        return match.group(1)
    return ''


# ============================================================
# 主分析函数
# ============================================================

def analyze_folder(folder_path):
    """分析文件夹内所有 CVS txt 文件。"""
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

        data = read_cv_data(filepath)
        if len(data) < 100:
            continue
        info['data'] = data
        all_files.append(info)

    print(f"共读取 {len(all_files)} 个 CVS 文件")

    # 2. 检测循环并积分（仅对 0.2V/s 的测试文件）
    results = []
    for f in all_files:
        data = f['data']
        reversals = detect_cycles(data)
        if len(reversals) < 3:
            print(f"  跳过 {f['file']}: 循环检测失败")
            continue

        last_fwd_start = reversals[-2]
        last_bwd_end = len(data)
        last_fwd = data[last_fwd_start:reversals[-1]]
        last_bwd = data[reversals[-1]:last_bwd_end]

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

        # 在 block 中找空白（泵4=0 且非VMS）
        baselines = [f for f in block if f['pumps'][3] == 0 and f['pumps'][0] != 6000]
        if not baselines:
            print(f"  Block {bi+1}: 未找到空白（泵4=0）")
            continue
        baseline = baselines[-1]

        # 找非空白文件（泵4>0），按泵4体积排序
        additions = [f for f in block if f['pumps'][3] > 0]
        additions.sort(key=lambda x: x['pumps'][3])

        if len(additions) < 2:
            print(f"  Block {bi+1}: 测量点不足（需要至少1个样品+1个加标）")
            continue

        # 泵4 的第一次非零体积 = 待测液（样品）体积
        V_sample = additions[0]['pumps'][3]

        # 读取底液体积（泵2）
        V_base = baseline['pumps'][1]
        if V_base == 0:
            V_base = 9400  # fallback

        print(f"  底液体积(V_base)={V_base} uL, 待测液体积(V_sample)={V_sample} uL")
        print(f"  空白: {baseline['file']}")
        print(f"  积分窗口 - 溶出: {baseline.get('ds_v_start',0):.4f}V → {baseline.get('ds_v_end',0):.4f}V")

        if RESPONSE_TYPE == 'charging':
            A0 = baseline['charge_area']
            area_label = '沉积'
            print(f"  空白沉积峰面积: {A0:.5f}")
        else:
            A0 = baseline['discharge_area']
            area_label = '溶出'
            print(f"  空白溶出峰面积: {A0:.5f}")

        std_points = []  # (C_eff, R) 用于标曲
        for s in additions:
            if RESPONSE_TYPE == 'charging':
                A = s['charge_area']
            else:
                A = s['discharge_area']
            R = A - A0

            V_pump4 = s['pumps'][3]  # 泵4 累积体积

            # 标准添加的有效浓度（扣除样品体积后的额外 STD）
            V_std_added = V_pump4 - V_sample  # 额外添加的 STD 体积
            V_total = V_base + V_pump4
            C_eff = V_std_added * C_STD / V_total if V_std_added > 0 else 0.0

            label = '样品' if V_std_added == 0 else f'+{V_std_added}uL STD'
            std_points.append({
                'file': s['file'],
                'V_pump4': V_pump4,
                'V_std_added': V_std_added,
                'C_eff': C_eff,
                'discharge_area': A,
                'R': R,
                'charge_area': s.get('charge_area', 0),
                'label': label
            })
            print(f"  {label} (泵4={V_pump4}uL): {area_label}面积={A:.5f}, R={R:.5f}, C_eff={C_eff:.4f}")

        # 选择参与拟合的点数（N_STD_POINTS=0 表示全部）
        if N_STD_POINTS > 0:
            # 样品点（第0个）+ 前 N_STD_POINTS 个加标点
            fit_points = std_points[:N_STD_POINTS + 1]
            print(f"  选择前 {N_STD_POINTS} 个加标点进行拟合（共 {len(std_points)-1} 个）")
        else:
            fit_points = std_points

        if len(fit_points) >= 2:
            C = np.array([p['C_eff'] for p in fit_points])
            R = np.array([p['R'] for p in fit_points])

            slope, intercept, r_value, p_value, std_err = stats.linregress(C, R)
            C_sample_eff = -intercept / slope if slope != 0 else 0

            # 换算回待测液（STD）原始浓度
            # C_sample_eff = C_x * V_sample / (V_base + V_sample)
            # → C_x = C_sample_eff * (V_base + V_sample) / V_sample
            C_x = abs(C_sample_eff) * (V_base + V_sample) / V_sample

            block_results.append({
                'block': bi + 1,
                'points': fit_points,
                'A0': A0,
                'V_base': V_base,
                'V_sample': V_sample,
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'C_sample_eff': C_sample_eff,
                'C_x': C_x,
            })

            print(f"\n  >>> 线性拟合:")
            print(f"      R = {slope:.5f} × C_eff + ({intercept:.5f})")
            print(f"      R² = {r_value**2:.4f}")
            print(f"      x截距 (C_sample_eff) = {C_sample_eff:.4f}")
            print(f"      待测液浓度 C_x = {C_x:.4f}")

    # 聚合分析
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
            # 聚合时使用平均的 V_base 和 V_sample
            avg_V_base = np.mean([r['V_base'] for r in block_results])
            avg_V_sample = np.mean([r['V_sample'] for r in block_results])
            C_x_p = abs(C_sample_eff_p) * (avg_V_base + avg_V_sample) / avg_V_sample
            pooled_result = {
                'block': 'All',
                'points': [{'C_eff': c, 'R': r, 'V_pump4': 0, 'V_std_added': 0,
                            'discharge_area': 0, 'charge_area': 0,
                            'file': 'pooled', 'label': 'pooled'}
                           for c, r in zip(all_C, all_R)],
                'A0': 0,
                'V_base': avg_V_base,
                'V_sample': avg_V_sample,
                'slope': slope_p,
                'intercept': intercept_p,
                'r_squared': r_p ** 2,
                'C_sample_eff': C_sample_eff_p,
                'C_x': C_x_p,
            }
            block_results.append(pooled_result)
            print(f"\n  >>> 聚合拟合 (All blocks pooled):")
            print(f"      R = {slope_p:.5f} × C_eff + ({intercept_p:.5f})")
            print(f"      R² = {r_p**2:.4f}")
            print(f"      C_x = {C_x_p:.4f}")

    return block_results, all_files


# ============================================================
# CSV 导出
# ============================================================

def export_csv(results, output_path):
    """导出浓度计算结果表为 CSV 文件。"""
    import csv

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)

        writer.writerow(['CVS 促进剂(A)浓度计算结果 — MLAT法'])
        writer.writerow([f'C_STD={C_STD} (STD浓度，也是待测液理论浓度), 响应类型={RESPONSE_TYPE}'])
        writer.writerow([])

        area_label = '溶出面积' if RESPONSE_TYPE == 'stripping' else '沉积面积'
        writer.writerow(['各测量点峰面积详情'])
        writer.writerow(['Block', '泵4(uL)', '加标量(uL)', 'C_eff', area_label, 'R值', '说明', '文件名'])

        individual_results = [r for r in results if r['block'] != 'All']
        for r in individual_results:
            for p in r['points']:
                area_val = p['discharge_area'] if RESPONSE_TYPE == 'stripping' else p['charge_area']
                writer.writerow([
                    f'Block {r["block"]}',
                    p['V_pump4'],
                    p['V_std_added'],
                    f'{p["C_eff"]:.4f}',
                    f'{area_val:.5f}',
                    f'{p["R"]:.5f}',
                    p['label'],
                    os.path.basename(p['file'])
                ])

        writer.writerow([])

        writer.writerow(['拟合结果汇总'])
        writer.writerow(['Block', 'V_base', 'V_sample', 'R²', '斜率', '截距', 'C_sample_eff', 'C_x (浓度)'])

        for r in results:
            label = f'Block {r["block"]}' if r['block'] != 'All' else 'Pooled'
            writer.writerow([
                label,
                f'{r.get("V_base", ""):.0f}' if isinstance(r.get('V_base'), (int, float)) else '',
                f'{r.get("V_sample", ""):.0f}' if isinstance(r.get('V_sample'), (int, float)) else '',
                f'{r["r_squared"]:.4f}',
                f'{r["slope"]:.5f}',
                f'{r["intercept"]:.5f}',
                f'{r["C_sample_eff"]:.4f}',
                f'{r["C_x"]:.4f}',
            ])

        C_x_values = [r['C_x'] for r in results if r['block'] != 'All']
        if len(C_x_values) > 1:
            mean_cx = np.mean(C_x_values)
            std_cx = np.std(C_x_values, ddof=1)
            rsd = std_cx / mean_cx * 100 if mean_cx != 0 else 0
            writer.writerow(['Mean', '', '', '', '', '', '', f'{mean_cx:.4f}'])
            writer.writerow(['SD', '', '', '', '', '', '', f'{std_cx:.4f}'])
            writer.writerow(['RSD%', '', '', '', '', '', '', f'{rsd:.2f}'])

    print(f"结果表已保存: {output_path}")


# ============================================================
# 绘图
# ============================================================

def plot_mlat(results, output_path=None):
    """绘制 MLAT 标曲图。"""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    individual_results = [r for r in results if r['block'] != 'All']
    pooled_results = [r for r in results if r['block'] == 'All']
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(individual_results), 1)))

    for i, r in enumerate(individual_results):
        C = np.array([p['C_eff'] for p in r['points']])
        R_data = np.array([p['R'] for p in r['points']])

        ax.scatter(C, R_data, color=colors[i], s=60, zorder=5, label=f'Block {r["block"]}')

        C_fit = np.linspace(min(C.min(), r['C_sample_eff'] - 0.5),
                            max(C.max(), 0), 100)
        R_fit = r['slope'] * C_fit + r['intercept']
        ax.plot(C_fit, R_fit, '--', color=colors[i], alpha=0.5, linewidth=1.2)

        ax.axvline(x=r['C_sample_eff'], color=colors[i], linestyle=':',
                    alpha=0.3, linewidth=0.8)

        ax.annotate(f'R²={r["r_squared"]:.3f}',
                     xy=(C.max(), R_data[-1]),
                     xytext=(C.max() * 0.6, R_data[-1] * 0.5 + R_data[0] * 0.5),
                     fontsize=11, color=colors[i],
                     alpha=0.8, ha='right')

    for r in pooled_results:
        C_all = np.array([p['C_eff'] for p in r['points']])
        ax.scatter(C_all, [p['R'] for p in r['points']],
                   color='black', s=15, zorder=4, alpha=0.4, label='_nolegend_')
        C_fit = np.linspace(min(C_all.min(), r['C_sample_eff'] - 0.5),
                            max(C_all.max(), 0), 100)
        R_fit = r['slope'] * C_fit + r['intercept']
        ax.plot(C_fit, R_fit, '-', color='black', linewidth=2.5, alpha=0.8,
                label=f"Pooled (C$_x$={r['C_x']:.2f})")
        ax.axvline(x=r['C_sample_eff'], color='black', linestyle='--',
                    alpha=0.6, linewidth=1)
        ax.annotate(f'C$_x$={r["C_x"]:.2f}',
                     xy=(r['C_sample_eff'], 0),
                     xytext=(r['C_sample_eff'] - 0.25, max(C_all)*0.05),
                     fontsize=12, color='black',
                     arrowprops=dict(arrowstyle='->', color='black'))
        ax.text(0.95, 0.05, f'Pooled R²={r["r_squared"]:.3f}',
                 transform=ax.transAxes, fontsize=12,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Effective concentration C_eff (std addition)')
    ax.set_ylabel('Response R = A - A0')
    ax.set_title('Standard Addition Calibration Curve — Accelerator (A)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.12)
    if output_path:
        plt.savefig(output_path, dpi=200)
        print(f"\nMLAT plot saved: {output_path}")


def plot_individual_mlat(result, output_path=None):
    """绘制单个 block 的 MLAT 标曲图。"""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    C = np.array([p['C_eff'] for p in result['points']])
    R_data = np.array([p['R'] for p in result['points']])

    ax.scatter(C, R_data, color='#2196F3', s=60, zorder=5, label='Data points')

    C_fit = np.linspace(min(C.min(), result['C_sample_eff'] - 0.5),
                        max(C.max(), 0), 100)
    R_fit = result['slope'] * C_fit + result['intercept']
    ax.plot(C_fit, R_fit, '-', color='#FF5722', linewidth=2, alpha=0.8,
             label=f"Fit: R={result['slope']:.4f}C+({result['intercept']:.4f})")

    ax.axvline(x=result['C_sample_eff'], color='#FF5722', linestyle='--',
                alpha=0.5, linewidth=1)
    ax.annotate(f'C$_x$={result["C_x"]:.2f}',
                 xy=(result['C_sample_eff'], 0),
                 xytext=(result['C_sample_eff'] - 0.35, max(R_data) * 0.7),
                 fontsize=13, color='#FF5722',
                 arrowprops=dict(arrowstyle='->', color='#FF5722'))

    ax.text(0.95, 0.05, f'R² = {result["r_squared"]:.4f}',
             transform=ax.transAxes, fontsize=14,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Effective concentration C_eff')
    ax.set_ylabel('Response R = A - A0')
    ax.set_title(f'Block {result["block"]} — Standard Addition (Accelerator A)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.12)
    if output_path:
        plt.savefig(output_path, dpi=200)
        print(f"Individual MLAT plot saved: {output_path}")


# ============================================================
# 结果输出
# ============================================================

def print_results_table(results):
    """打印各 block 的详细结果表。"""
    print("\n" + "="*80)
    print("促进剂(A)浓度计算结果 — MLAT法")
    print("="*80)

    print(f"\n{'Block':<8} {'R²':<10} {'斜率':<12} {'截距':<12} {'C_sample_eff':<15} {'C_x (浓度)':<15}")
    print("-"*75)

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
        print("-"*75)
        print(f"{'Mean':<8} {'':<10} {'':<12} {'':<12} {'':<15} {mean_cx:<15.4f}")
        print(f"{'SD':<8} {'':<10} {'':<12} {'':<12} {'':<15} {std_cx:<15.4f}")
        print(f"{'RSD%':<8} {'':<10} {'':<12} {'':<12} {'':<15} {rsd:<15.2f}")

    print("\n" + "="*90)
    print("各测量点峰面积详情")
    print("="*90)
    print(f"{'Block':<8} {'泵4(uL)':<10} {'加标(uL)':<10} {'C_eff':<10} {'溶出面积':<12} {'R值':<12} {'说明':<12} {'文件':<25}")
    print("-"*90)
    for r in results:
        if r['block'] == 'All':
            continue
        for p in r['points']:
            print(f"{r['block']:<8} {p['V_pump4']:<10} {p['V_std_added']:<10} {p['C_eff']:<10.4f} "
                  f"{p['discharge_area']:<12.5f} {p['R']:<12.5f} "
                  f"{p['label']:<12} {os.path.basename(p['file'])[:23]:<23}")


# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    import sys
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    default_path = r"D:\文档\Python Documents\2026\Cu\5#\26-05\box-A\txt_box"

    folder = default_path
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            folder = arg
            break

    if not os.path.isdir(folder):
        print(f"错误: 文件夹不存在 - {folder}")
        sys.exit(1)

    print(f"分析文件夹: {folder}")
    print(f"参数: C_STD={C_STD}, RESPONSE_TYPE={RESPONSE_TYPE}")
    print(f"泵配置: 1=VMS(活化), 2=底液, 4=待测液(STD)+标准添加, 3/5/6=空")

    results, all_files = analyze_folder(folder)

    if not results:
        print("未生成有效结果，请检查数据格式")
        sys.exit(1)

    print_results_table(results)

    parent_dir = os.path.dirname(folder)
    csv_path = os.path.join(parent_dir, '计算结果.csv')
    export_csv(results, csv_path)

    plot_path = os.path.join(parent_dir, 'MLAT_plot.png')
    plot_mlat(results, plot_path)

    for r in results:
        if r['block'] == 'All':
            continue
        individual_path = os.path.join(parent_dir, f'MLAT_plot_Block{r["block"]}.png')
        plot_individual_mlat(r, individual_path)

    print("\n分析完成!")
