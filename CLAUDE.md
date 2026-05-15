# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CVS (Cyclic Voltammetry Stripping) 整平剂/促进剂浓度自动计算脚本。基于标准加入法（MLAT），通过积分 CV 曲线计算电镀液中的添加剂浓度。

- `analyze_cvs.py` — 测 **L（整平剂）**浓度
- `analyze_a_cvs.py` — 测 **A（促进剂）**浓度

## Python 环境

虚拟环境位于 `../.venv`（项目父目录），需使用该环境的 Python 运行：

```bash
# 测 L（整平剂）
../.venv/Scripts/python.exe analyze_cvs.py [数据文件夹路径]

# 测 A（促进剂）
../.venv/Scripts/python.exe analyze_a_cvs.py [数据文件夹路径]
```

## Code Architecture

### analyze_cvs.py — 测 L（整平剂）

**泵配置**: P1=VMS, P2=底液, P3=待测液, P4=标准液L, P5=促进剂, P6=抑制剂
**响应值**: `R = A₀ - A`（整平剂抑制反应，空白峰面积最大，R 恒正）
**配置**: `C_STD=50`, `V_TEST=5000`, `V_BASE=5000`

### analyze_a_cvs.py — 测 A（促进剂）

**泵配置**: P1=VMS, P2=底液, P3=空, P4=待测液+标准添加（同一泵）, P5=空, P6=空
**响应值**: `R = A - A₀`（促进剂增强反应，峰面积随浓度增大，R 恒正）
**配置**: `C_STD=23`（STD 中 A 浓度为 23）

与测 L 的关键区别：
- 泵4 双重角色：第一次非零泵4体积 = 待测液，后续累积体积 = 标准添加
- `V_base` 从泵2 文件名自动读取，不依赖全局常量
- 无泵3 待测液

### 共用实现

两个脚本各自包含完整的代码（刻意不抽取共享库，保持独立可运行）。
底层的文件解析、循环检测、积分算法完全一致：

**数据解析（`parse_filename`, `read_cv_data`）**
- 文件名格式：`CV_{P1}-{P2}-{P3}-{P4}-{P5}-{P6}_{扫速}_{转速}_{日期}.txt`
- 数据文件跳过前 5 行表头，每行 `电位 电流`

**循环检测与积分（`detect_cycles`, `integrate_stripping`, `integrate_deposition`）**
- 0.2V/s 数据 2 个循环，0.1V/s 数据 3 个循环，每个半圈 851 点
- 溶出峰积分（放电）：从电流 0 过零点（负→正）到 0.5V
- 沉积积分（充电）：从电流明显下降处开始，经转折点，到溶出峰后电流归零结束

**分组与标准加入法（`group_by_block`, `analyze_folder`）**
- 按 VMS（P1=6000）标记分 block
- 线性拟合 R vs C_eff，x 截距（R=0 时的 C_eff）换算出待测液浓度 C_x
- 聚合分析（pooled）：所有 block 数据点统一拟合

**输出（`export_csv`, `plot_mlat`, `plot_individual_mlat`, `print_results_table`）**
- 输出到数据文件夹的**父目录**下
- `计算结果.csv`、`MLAT_plot.png`、`MLAT_plot_BlockN.png`

**关键依赖**
- `numpy`, `scipy.stats`, `matplotlib`（Times New Roman 字体，Agg 后端）

## GitHub

- 仓库：https://github.com/CuSO4-5H2O-0/cvs-analyzer
