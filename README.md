# PINN Subduction Visualization

基于物理信息神经网络 (PINN) 与普通前馈神经网络 (FNN) 的俯冲板片温度场预测可视化。

## 功能

- 输入地球动力学参数（迭代时间、俯冲板片年龄、俯冲速度等）
- 同时运行 FNN 与 PINN 两个模型推理
- 并排展示两个模型的温度场预测结果，共享色标
- 支持下载对比图

## 技术栈

- **UI**：Streamlit
- **推理**：PyTorch (CPU)
- **绘图**：matplotlib（polar 色标，0~1320 K）
- **托管**：Streamlit Community Cloud（免费）

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
# 访问 http://localhost:8501
```

## 部署到 Streamlit Community Cloud

1. 将本仓库推送到 GitHub（公开仓库）
2. 访问 https://share.streamlit.io
3. 登录 GitHub，选择仓库
4. 主文件设为 `app.py`
5. 自动构建部署，获得公网访问链接
