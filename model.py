"""PINN 模型推理模块。

加载训练好的全连接网络（FNN）权重，对俯冲板片地球动力学模拟的
网格点进行推理，输出温度场预测结果。

模型权重以 GPU 保存，通过 map_location 自动适配当前设备（CPU/GPU）。
"""

import os

import numpy as np
import torch
import torch.nn as nn

import config


# 设备配置：优先使用 GPU，回退到 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FNN(nn.Module):
    """最基础的线性神经网络（多层感知机）。

    输入: 时间 t、热流 H、倾角 d、年龄 Y、俯冲速度 v 等 7 维特征
          （深度 D、长度 L 暂不作为输入；当前数据深度 600km，长度 660km）
    输出: 温度场，分辨率 nx × nz

    网络结构: 输入层 → depth 个隐藏层 → 输出层
              7→100→(100→100)×depth→1，共 depth+2 层 Linear。
              最后一个隐藏层后用 Sigmoid（因能量守恒公式含温度对时空的
              二阶导，否则二阶导为 0），其余隐藏层用 LeakyReLU。
    """

    def __init__(self, in_features: int = 7, out_features: int = 1,
                 hidden: int = 100, depth: int = 12):
        super().__init__()
        # 输入层
        layers = [nn.Linear(in_features, hidden), nn.LeakyReLU(inplace=True)]
        # depth 个隐藏层：前 depth-1 个接 LeakyReLU，最后一个接 Sigmoid
        for i in range(depth):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.LeakyReLU(inplace=True) if i < depth - 1
                          else nn.Sigmoid())
        # 输出层
        layers.append(nn.Linear(hidden, out_features))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def load_model(model_path: str) -> "FNN":
    """加载模型权重并返回 eval 模式的网络实例（可缓存复用）。"""
    net = FNN().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    return net


def test(
    model_path: str,
    target_name: str = "TempHeat0645.0Age02Vel01.11350",
    temp_heat: float = 217.5,
    age: float = 60.0,
    dip: float = 45.0,
    age_plate: float = 30.0,
    vel: float = 3.17,
    adapt_mode: str = "default",
    root_dir: str = ".",
    result_dir: str = "./result",
    net: "FNN" = None,
):
    """加载模型并对网格点进行推理，输出预测结果。

    参数:
        model_path:  模型权重文件路径（.pt/.pth，GPU 保存亦可）
        target_name: 验证样本名称，用于定位真实数据文件与命名输出
        temp_heat:   温度热源参数
        age:         年龄参数
        dip:         俯冲角度
        age_plate:   板片年龄
        vel:         速度参数
        adapt_mode:  自适应模式标识，用于结果目录命名
        root_dir:    项目根目录，用于定位验证数据
        result_dir:  结果输出目录
        net:         可选，预加载的网络实例（传入则跳过 model_path 加载）

    返回:
        dict: fake（预测值）、real（真值归一化，无真值时为 None）、
              target_name、out_file（预测结果文件路径）
    """
    # 加载模型权重（map_location 自动把 GPU 权重映射到当前设备）
    if net is None:
        net = load_model(model_path)

    # 输出目录
    test_dir = os.path.join(result_dir, f"pifnn_{adapt_mode}_test")
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, f"pifnn_{adapt_mode}_convert"), exist_ok=True)

    with torch.no_grad():
        # 构造网格输入：x 方向 0~660，z 方向 0~600，步长 6
        # 每行 7 维: [x, 600-z, temp_heat, age, dip, age_plate, vel]
        xs = range(0, config.GRID_X_LEN + 1, config.GRID_STEP)
        zs = range(0, config.GRID_Z_LEN + 1, config.GRID_STEP)
        arr = [[x, config.GRID_Z_LEN - z, temp_heat, age, dip, age_plate, vel]
               for x in xs for z in zs]

        input_tensor = torch.from_numpy(np.array(arr, dtype=float)).float().to(device)

        # 模型推理（输出 1 维）
        fake_img = net(input_tensor)
        fake = fake_img.cpu().numpy()[:, 0]

        # 保存预测结果
        out_file = os.path.join(test_dir, f"fake{target_name}.txt")
        np.savetxt(out_file, fake, fmt="%f", delimiter=" ")

        # 读取真实数据用于对比（若存在）
        real_img = None
        real_path = os.path.join(root_dir, "valid", "output", target_name)
        if os.path.exists(real_path):
            real_img = np.loadtxt(real_path, delimiter=',')[:, 0]
            real_img = np.clip(real_img, 0.0, 1320.0) / 1320.0

        return {
            "fake": fake,
            "real": real_img,
            "target_name": target_name,
            "out_file": out_file,
        }


if __name__ == "__main__":
    # 命令行直接运行时的简易测试入口
    import argparse

    parser = argparse.ArgumentParser(description="PINN 模型推理")
    parser.add_argument("model_path", help="模型权重文件路径")
    parser.add_argument("--root_dir", default=".", help="项目根目录")
    args = parser.parse_args()

    result = test(args.model_path, root_dir=args.root_dir)
    print(f"预测样本数: {len(result['fake'])}")
    print(f"结果已保存至: {result['out_file']}")
