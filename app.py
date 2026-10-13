"""Streamlit 应用 - 俯冲板片地球动力学模拟可视化。

Streamlit Community Cloud 入口。
用户输入控制参数 → FNN/PINN 双模型推理 → 并排展示温度场对比图。
"""

import os
import time
from collections import OrderedDict

import streamlit as st

import config
import plotting

# model 依赖 torch，缺失时仍允许 UI 访问，仅推理报错
try:
    import model
    MODEL_AVAILABLE = True
    MODEL_ERROR = None
except ImportError as e:
    MODEL_AVAILABLE = False
    MODEL_ERROR = str(e)


def get_net(model_path):
    """获取（必要时加载并缓存）模型网络实例。Streamlit cache_resource 自动跨请求复用。"""
    return model.load_model(model_path)


def discover_models():
    """扫描模型目录，返回 OrderedDict {"FNN": 路径, "PINN": 路径}。"""
    found = OrderedDict()
    if not os.path.isdir(config.MODEL_DIR):
        return found
    exts = (".pt", ".pth", ".pkl")
    for fname in sorted(os.listdir(config.MODEL_DIR)):
        if not fname.endswith(exts):
            continue
        mtype = parse_model_info(fname)["type"]
        if mtype not in found:
            found[mtype] = os.path.join(config.MODEL_DIR, fname)
    return found


def parse_model_info(filename):
    """解析模型文件名，返回类型与物理约束权重信息。

    命名规则: pifnn_{w1}_{w2}_{w3}_{w4}_channel{c}.pth
      物理约束权重全为 0 → FNN；含非 0 → PINN。
    """
    base = filename.rsplit(".", 1)[0]
    parts = base.split("_")
    info = {"name": filename, "type": "FNN", "channel": 1, "weights": []}
    if len(parts) >= 6 and parts[0] == "pifnn":
        weights = parts[1:5]
        try:
            floats = [float(w) for w in weights]
            info["weights"] = weights
            info["type"] = "PINN" if any(v > 0 for v in floats) else "FNN"
        except ValueError:
            pass
        ch = parts[5]
        if ch.startswith("channel"):
            try:
                info["channel"] = int(ch.replace("channel", ""))
            except ValueError:
                pass
    return info


def run_simulation(temp_heat, age, dip, age_plate, vel):
    """推理 + 绘图。返回 (图片路径, 状态文本)。"""
    if not MODEL_AVAILABLE:
        return None, f"Model inference unavailable (torch missing): {MODEL_ERROR}"

    model_paths = discover_models()
    if not model_paths:
        return None, "No model files found in models/ directory"

    params = {
        "temp_heat": float(temp_heat),
        "age": float(age),
        "dip": float(dip),
        "age_plate": float(age_plate),
        "vel": float(vel) * 3.17,  # 用户输入值 × 3.17 为模型实际速度
    }
    target_name = "TempHeat0645.0Age02Vel01.11350"

    preds = OrderedDict()
    count = 0
    try:
        for label, path in model_paths.items():
            net = get_net(path)
            result = model.test(
                model_path=path,
                target_name=target_name,
                adapt_mode=label.lower(),
                root_dir=config.DATA_DIR,
                result_dir=config.RESULT_DIR,
                net=net,
                **params,
            )
            preds[label] = result["fake"]
            count = len(result["fake"])
    except Exception as e:
        return None, f"Inference failed: {e}"

    # 绘图到文件
    stamp = int(time.time())
    out_png = os.path.join(config.RESULT_DIR, f"models_{stamp}.png")
    plotting.plot_models_comparison(preds, out_png, title=target_name, params=params)

    labels = " / ".join(preds.keys())
    return out_png, f"Done: {labels}, {count} grid points."


# ============ Streamlit UI ============

st.set_page_config(page_title="PINN Subduction Visualization", page_icon="🌋", layout="wide")

st.title("Predicting Thermal Structure of Plate Subduction by PINN and FNN Models")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("Parameters")
    temp_heat = st.number_input("Iteration Time (Ma)", value=config.DEFAULT_PARAMS["temp_heat"])
    age = st.number_input("Overriding Plate Age (Ma)", value=config.DEFAULT_PARAMS["age"])
    dip = st.number_input("Subduction Angle (°)", value=config.DEFAULT_PARAMS["dip"])
    age_plate = st.number_input("Subducting Plate Age (Ma)", value=config.DEFAULT_PARAMS["age_plate"])
    vel = st.number_input("Subduction Velocity (cm/yr)", value=config.DEFAULT_PARAMS["vel"])
    btn = st.button("Run Simulation", type="primary")

with col2:
    st.header("Result")
    if btn:
        with st.spinner("Running FNN and PINN inference..."):
            img_path, status = run_simulation(temp_heat, age, dip, age_plate, vel)
        if img_path:
            st.image(img_path, caption="FNN vs PINN Comparison", use_container_width=True)
            st.success(status)
            # 下载按钮
            with open(img_path, "rb") as f:
                st.download_button(
                    label="Download Image",
                    data=f.read(),
                    file_name=os.path.basename(img_path),
                    mime="image/png",
                )
        else:
            st.error(status)
    elif "last_image" in st.session_state:
        st.image(st.session_state.last_image, use_container_width=True)
        if "last_status" in st.session_state:
            st.success(st.session_state.last_status)
    else:
        st.info("Click 'Run Simulation' to generate the temperature field comparison.")

# 点击后缓存结果到 session_state（页面其他交互时不丢失）
if btn and img_path:
    st.session_state.last_image = img_path
    st.session_state.last_status = status
