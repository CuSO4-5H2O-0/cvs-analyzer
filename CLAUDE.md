# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CVS (Cyclic Voltammetry Stripping) 整平剂浓度自动计算脚本。基于标准加入法，通过积分 CV 曲线计算电镀液中的整平剂浓度。

## Commands

```bash
# 运行分析
python analyze_cvs.py [数据文件夹路径]

# 无参数时使用默认路径
python analyze_cvs.py
```

## Code Architecture

单一脚本 `analyze_cvs.py`，按功能分为以下模块：

### 配置参数
- `C_STD` — 标准液整平剂浓度（默认 50）
- `V_TEST` / `V_BASE` — 待测液 / 底液体积（uL）
- `RESPONSE_TYPE` — `'stripping'`（溶出峰）或 `'charging'`（沉积峰）
- `N_STD_POINTS` — 用于拟合的标准添加点数（0=全部）

### 数据解析（`parse_filename`, `read_cv_data`）
- 文件名格式：`CV_{P1}-{P2}-{P3}-{P4}-{P5}-{P6}_{扫速}_{转速}_{日期}.txt`
- 泵定义：P1=VMS(仅活化电极，不参与测试), P2=底液(体积可能变化), P3=待测液(体积可能变化), P4=标准液L(整平剂), P5=促进剂, P6=抑制剂
- 数据文件跳过前 5 行表头，每行 `电位 电流`

### 循环检测与积分（`detect_cycles`, `integrate_stripping`, `integrate_deposition`）
- 0.2V/s 数据 2 个循环，0.1V/s 数据 3 个循环，每个半圈 851 点
- **溶出峰积分（放电）**：从电流 0 过零点（负→正）到 0.5V
- **沉积积分（充电）**：从电流明显下降处开始，经转折点，到溶出峰后电流归零结束

### 分组与标准加入法（`group_by_block`, `analyze_folder`）
- 按 VMS（P1=6000）标记分 block，每个 block 含空白 + 4 个标准添加（0,100,200,300 uL）
- 响应值 R = A₀ - A，线性拟合 R vs C_eff，x 截距得待测液浓度
- 聚合分析（pooled）：所有 block 数据点统一拟合

### 输出（`export_csv`, `plot_mlat`, `plot_individual_mlat`, `print_results_table`）
- 终端结果表 + 各点峰面积详情
- `计算结果.csv` — 包含各点数据和拟合汇总
- `MLAT_plot.png` — 所有 block 叠加的标曲图（R vs C_eff）
- `MLAT_plot_BlockN.png` — 每个 block 的单独标曲图

### 关键依赖
- `numpy` — 数值计算
- `scipy.stats` — 线性回归
- `matplotlib` — 绘图（Times New Roman 字体，Agg 后端）

## GitHub

- 仓库：https://github.com/CuSO4-5H2O-0/cvs-analyzer
