"""项目配置：路径、默认参数、服务选项。"""

import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型权重目录
MODEL_DIR = os.path.join(BASE_DIR, "models")

# 验证数据目录
DATA_DIR = os.path.join(BASE_DIR, "data")

# 结果输出目录
RESULT_DIR = os.path.join(BASE_DIR, "result")

# 静态资源目录
STATIC_DIR = os.path.join(BASE_DIR, "static")

# 推理网格参数
# GRID_X_LEN/Z_LEN 为物理长度（km），range(0, LEN+1, STEP) 生成 nx×nz 个点
GRID_X_LEN = 660
GRID_Z_LEN = 600
GRID_STEP = 6

# 温度场显示配置
# 模型输出为 0~1 归一化值，乘以 TEMP_SCALE 还原为开尔文温度（与训练归一化上限一致）
# 物理上地幔温度截断到 TEMP_CLAMP，色标范围 0~CPT_MAX（参考 plot1NN.sh: makecpt -T0/1320）
TEMP_SCALE = 1320        # 还原系数（×1320）
TEMP_CLAMP = 1300        # 物理温度截断上限（K）
CPT_MAX = 1320           # 色标范围上限（K）
CMAP = "polar"           # matplotlib/GMT 色标

# 默认控制参数（俯冲板片地球动力学）
DEFAULT_PARAMS = {
    "temp_heat": 217.5,
    "age": 60.0,
    "dip": 45.0,
    "age_plate": 30.0,
    "vel": 1.0,
}

# 默认模型文件名（放在 models/ 目录下）
DEFAULT_MODEL_NAME = "pifnn_model.pt"

# GMT 可执行文件路径，若 GMT 已加入 PATH 可留空
GMT_BIN = ""

# Flask 服务配置
# HF Spaces 默认端口 7860；本地可设 PORT=5000
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 7860))
DEBUG = False
