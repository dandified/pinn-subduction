"""绘图模块。

优先使用 GMT（Generic Mapping Tools）绘制俯冲板片温度场；
若 GMT 不可用，自动降级为 matplotlib 绘图，保证流程可用。

坐标约定（参考 generate_tempfile.py / plot1NN.sh）：
  - 原点在左上角，x 轴向右（0~660 km），z 轴向下表示深度（0~600 km）。
  - 模型输出为 0~1 归一化值，乘以 1320 还原为开尔文，物理截断到 1300 K。
  - 色标 polar（蓝-白-红），范围 0~1320 K。
  - 叠加等温线（200 K 间隔，标注 400 K，仅 400~1200 K 范围）。
"""

import os
import shutil
import subprocess

import numpy as np

import config

# matplotlib 无 GUI 后端，适合服务端出图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# 网格维度：range(0, LEN+1, STEP) 的点数
_NX = config.GRID_X_LEN // config.GRID_STEP + 1  # 111
_NZ = config.GRID_Z_LEN // config.GRID_STEP + 1  # 101

# 图像坐标范围 [left, right, bottom, top]，top=0 使原点落在左上角
_EXTENT = [0, config.GRID_X_LEN, config.GRID_Z_LEN, 0]

# 等温线参数（参考 plot1NN.sh: grdcontour -C200 -A400 -L400/1200）
_CONTOUR_LEVELS = list(range(400, 1201, 200))  # 400,600,800,1000,1200

# GMT 色标名 -> matplotlib 等价色图
# GMT polar 为 蓝(低)->白(中)->红(高) 的发散色图，等价于 matplotlib bwr
_MPL_CMAP_MAP = {"polar": "bwr"}


def reshape_grid(flat: np.ndarray, nx: int = None, nz: int = None) -> np.ndarray:
    """将一维预测结果重塑为二维网格 (nx, nz)。

    模型输出顺序为 x 外层、z 内层，故 reshape 为 (nx, nz)。
    grid[i][j] 对应物理坐标 (x=6i, z=600-6j)：j=0 为底部(z=600)，j=nz-1 为顶部(z=0)。
    """
    if nx is None:
        nx = _NX
    if nz is None:
        nz = _NZ
    return flat.reshape(nx, nz)


def _to_kelvin(flat: np.ndarray) -> np.ndarray:
    """归一化温度（0~1）转换为开尔文：×1320 还原，截断到 0~1300 K。

    参考 generate_tempfile.py: clamp(v * 1320.0, 0.0, 1300.0)。
    """
    return np.clip(np.asarray(flat, dtype=np.float64) * config.TEMP_SCALE,
                   0.0, config.TEMP_CLAMP)


def _mpl_cmap() -> str:
    return _MPL_CMAP_MAP.get(config.CMAP, config.CMAP)


def _grid_for_imshow(flat: np.ndarray) -> np.ndarray:
    """转换为 imshow 所需的 (nz, nx) 数组，行 0 = 顶部(z=0)。

    grid[i][j] 中 j=0 对应 z=600(底部)，j=nz-1 对应 z=0(顶部)。
    transpose 后 row=j，需 flipud 使 row 0 = 顶部。
    """
    grid = reshape_grid(_to_kelvin(flat))  # (nx, nz)
    return np.flipud(grid.T)               # (nz, nx), row0=顶部


def _imshow(ax, flat, title):
    """在指定 ax 上绘制单张温度场（含等温线），统一色标与坐标约定。"""
    data = _grid_for_imshow(flat)
    im = ax.imshow(
        data, aspect="auto", origin="upper", extent=_EXTENT,
        cmap=_mpl_cmap(), vmin=0, vmax=config.CPT_MAX,
    )
    # 等温线（参考 plot1NN.sh: grdcontour -C200 -A400 -L400/1200）
    x = np.linspace(0, config.GRID_X_LEN, _NX)
    z = np.linspace(0, config.GRID_Z_LEN, _NZ)
    cs = ax.contour(x, z, data, levels=_CONTOUR_LEVELS,
                    colors="black", linewidths=0.8)
    ax.clabel(cs, levels=[400, 800, 1200], fmt="%d", fontsize=7)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Horizontal distance (km)", fontsize=13)
    ax.set_ylabel("Depth (km)", fontsize=13)
    ax.tick_params(labelsize=11)
    return im


def _gmt_cmd(name: str) -> str:
    return f"{config.GMT_BIN}{name}" if config.GMT_BIN else name


def gmt_available() -> bool:
    """检测 GMT 是否可用。"""
    return shutil.which(_gmt_cmd("gmt")) is not None


# ------------------------------------------------------------------
# matplotlib 绘图
# ------------------------------------------------------------------

def _plot_temperature_mpl(flat, out_png, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = _imshow(ax, flat, title)
    fig.colorbar(im, ax=ax, label="Temperature (K)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def _plot_comparison_mpl(fake_flat, real_flat, out_png, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, flat, name in [
        (axes[0], fake_flat, f"{title} - Prediction"),
        (axes[1], real_flat, f"{title} - Reference"),
    ]:
        im = _imshow(ax, flat, name)
        fig.colorbar(im, ax=ax, label="Temperature (K)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def _format_params(p):
    """格式化控制参数为两行文本，显示在图顶部。"""
    line1 = (f"Iteration Time: {p['temp_heat']} Ma | "
             f"Overriding Plate Heat Flux: {p['age']} mW/m² | "
             f"Subduction Angle: {p['dip']}°")
    line2 = (f"Slab Age: {p['age_plate']} Ma | "
             f"Slab Velocity: {p['vel'] / 3.17:.1f} cm/yr")
    return line1 + "\n" + line2


def _plot_models_comparison_mpl(preds, out_png, title, params=None):
    """preds: 有序字典 {模型标签: 一维归一化预测结果}，并排绘制。

    标题只显示模型标签（FNN / PINN），两张图共用底部 colorbar。
    若传入 params，则在图顶显示一行控制参数。
    """
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 7.5), squeeze=False)
    # 顶部预留空间给两行参数
    fig.subplots_adjust(top=0.82)
    axes = axes[0]
    im = None
    for ax, (label, flat) in zip(axes, preds.items()):
        im = _imshow(ax, flat, label)
    # 共用底部 colorbar
    fig.colorbar(im, ax=list(axes), location="bottom",
                 shrink=0.7, pad=0.10, label="Temperature (K)")
    # 顶部参数行（两行，放大字号）
    if params:
        fig.suptitle(_format_params(params), fontsize=16, y=0.96)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ------------------------------------------------------------------
# GMT 绘图（参考 plot1NN.sh）
# ------------------------------------------------------------------

def _to_grd(grid_kelvin, out_grd):
    """将二维温度数组（K）写为 GMT 网格文件（.grd）。

    参考 plot1NN.sh: xyz2grd -R0/660/0/600 -I6/6。
    """
    nx, nz = grid_kelvin.shape
    bin_file = out_grd + ".bin"
    grid_kelvin.T.astype(np.float32).tofile(bin_file)
    subprocess.run([
        _gmt_cmd("xyz2grd"), bin_file, "-G", out_grd,
        f"-R0/{config.GRID_X_LEN}/0/{config.GRID_Z_LEN}",
        f"-I{config.GRID_STEP}/{config.GRID_STEP}", "-ZTL",
    ], check=True)
    os.remove(bin_file)
    return out_grd


def _write_polar_cpt(cpt_file):
    """生成 polar 风格 CPT（蓝-白-红），范围 0~CPT_MAX。

    参考 plot1NN.sh: makecpt -Cpolar -T0/1320/10 -Z。
    GMT 内置 polar 带 SOFT_HINGE（中点 0），正值区间只剩白->红，
    故手写三节点保证全范围蓝->白->红，与 matplotlib bwr 一致。
    """
    mid = config.CPT_MAX / 2
    lines = [
        "# auto-generated polar-like cpt (blue-white-red)",
        "# COLOR_MODEL = RGB",
        f"0\t0/0/255\t{mid:g}\t255/255/255",
        f"{mid:g}\t255/255/255\t{config.CPT_MAX:g}\t255/0/0",
        "B\t0/0/255",
        "F\t255/0/0",
        "N\t128/128/128",
    ]
    with open(cpt_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return cpt_file


def _gmt_panel(grd_file, cpt_file, out_png, mode, x_off, y_off):
    """绘制单个 GMT 面板（grdimage + grdcontour）。

    参考 plot1NN.sh:
      grdimage -JX6.6/-6.0 -R0/660/0/600 -C$cpt -B... -K
      grdcontour -C200 -A400 -L400/1200 -O -K
    mode: "new" (第一幅, >), "append" (后续, >>)
    """
    region = f"-R0/{config.GRID_X_LEN}/0/{config.GRID_Z_LEN}"
    # -JX 中 z 用负号使 z 轴向下为正（原点左上）
    proj = f"-JX{config.GRID_X_LEN / 100}c/-{config.GRID_Z_LEN / 100}c"
    frame = '-Ba100f50:"Horizontal distance (km)":/a100f50:"Depth (km)"::."":WeSn'
    redirect = ">" if mode == "new" else ">>"
    flag = "-K" if mode == "new" else "-K -O"

    subprocess.run(
        f'{_gmt_cmd("grdimage")} {grd_file} {region} {proj} -C{cpt_file} -Q '
        f'{frame} -X{x_off} -Y{y_off} {flag} {redirect} {out_png}',
        shell=True, check=True,
    )
    # 等温线
    subprocess.run(
        f'{_gmt_cmd("grdcontour")} {grd_file} {region} {proj} -C200 -A400 '
        f'-Wathick -Wcthick -L400/1200 -O {redirect} {out_png}',
        shell=True, check=True,
    )


def _plot_temperature_gmt(flat, out_png, title):
    grd_file, cpt_file = out_png + ".grd", out_png + ".cpt"
    grid = reshape_grid(_to_kelvin(flat))
    _to_grd(grid, grd_file)
    _write_polar_cpt(cpt_file)
    _gmt_panel(grd_file, cpt_file, out_png, "new", "3c", "5c")
    subprocess.run(
        f'{_gmt_cmd("psscale")} -C{cpt_file} -D8c/8.5c/16c/0.2h '
        f'-B200:"Temperature (K)": -O >> {out_png}',
        shell=True, check=True,
    )
    for f in (grd_file, cpt_file):
        if os.path.exists(f):
            os.remove(f)
    return out_png


def _plot_models_comparison_gmt(preds, out_png, title):
    """多个模型横向并排（参考 plot1NN.sh 风格）。"""
    cpt_file = out_png + ".cpt"
    _write_polar_cpt(cpt_file)
    tmp_grds = []
    for i, (label, flat) in enumerate(preds.items()):
        grd_file = f"{out_png}.{i}.grd"
        _to_grd(reshape_grid(_to_kelvin(flat)), grd_file)
        tmp_grds.append(grd_file)
        mode = "new" if i == 0 else "append"
        x_off = "3c" if i == 0 else "9c"
        _gmt_panel(grd_file, cpt_file, out_png, mode, x_off, "5c")
    # 末尾 colorbar
    subprocess.run(
        f'{_gmt_cmd("psscale")} -C{cpt_file} -D8c/8.5c/16c/0.2h '
        f'-B200:"Temperature (K)": -O >> {out_png}',
        shell=True, check=True,
    )
    for f in tmp_grds + [cpt_file]:
        if os.path.exists(f):
            os.remove(f)
    return out_png


def _plot_comparison_gmt(fake_flat, real_flat, out_png, title):
    return _plot_models_comparison_gmt(
        {"Prediction": fake_flat, "Reference": real_flat}, out_png, title
    )


# ------------------------------------------------------------------
# 统一入口：自动选择 GMT 或 matplotlib
# ------------------------------------------------------------------

def plot_temperature(flat_data, out_png, title="Subduction Temperature Field"):
    """绘制单张温度场图（GMT 优先，不可用则 matplotlib）。"""
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    if gmt_available():
        return _plot_temperature_gmt(flat_data, out_png, title)
    return _plot_temperature_mpl(flat_data, out_png, title)


def plot_comparison(fake_flat, real_flat, out_png, title="Prediction vs. Reference"):
    """绘制预测与真值并排对比图（GMT 优先，不可用则 matplotlib）。"""
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    if gmt_available():
        return _plot_comparison_gmt(fake_flat, real_flat, out_png, title)
    return _plot_comparison_mpl(fake_flat, real_flat, out_png, title)


def plot_models_comparison(preds, out_png, title="FNN vs. PINN", params=None):
    """并排绘制多个模型（如 FNN / PINN）的温度场预测。"""
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    preds = dict(preds)
    if gmt_available():
        return _plot_models_comparison_gmt(preds, out_png, title)
    return _plot_models_comparison_mpl(preds, out_png, title, params)


if __name__ == "__main__":
    backend = "GMT" if gmt_available() else "matplotlib"
    print(f"使用后端: {backend}")
    demo = {
        "FNN": np.clip(np.random.rand(_NX * _NZ) * 0.9 + 0.05, 0, 1),
        "PINN": np.clip(np.random.rand(_NX * _NZ) * 0.9 + 0.05, 0, 1),
    }
    plot_models_comparison(demo, "./result/test_models.png")
