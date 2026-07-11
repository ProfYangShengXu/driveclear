# DriveClear — 驾驶视界增强系统

> **ECE4512 数字图像处理 · 2025 Final Project**
>
> 作者：韩廷欣 (124090173) · 陈廷俊 (124090090)

行车视频的「全能滤镜」—— 自动识别并修复雾霾 / 眩光 / 低光照，让每帧画面都接近晴日白天的质量。

---

## 目录

- [解决的问题](#-解决的问题)
- [算法管线](#-算法管线)
- [效果概览](#-效果概览)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [API 接口](#-api-接口)
- [评估与消融](#-评估与消融)
- [技术栈](#-技术栈)
- [时间线](#-时间线)

---

## 🎯 解决的问题

行车记录仪和 ADAS 摄像头在恶劣天气下画质断崖式下降，而现有方法只处理单一退化：

| 退化类型 | 现有方法局限 | DriveClear 方案 |
|----------|-------------|----------------|
| 🌫️ 雾霾 | DCP 单帧去雾无法用时序信息，雾重区域色彩失真 | 时域雾层估计 + DCP 自适应融合 |
| ☀️ 镜头眩光 | 传统方法难以去除光晕，易产生黑色伪影 | 亮通道检测 + 局部亮度抑制 + 时域插值 |
| 🌙 低光照 | CLAHE/Retinex 单一增强过度放大噪声 | Retinex 分解 + 照明层伽马 + 噪声感知增益控制 |
| 🌪️ **混合退化** | 现有工具只能处理一种，无法应对同时出现的多种退化 | 可插拔流水线 + 帧级退化检测自动调度 |

**核心创新：**
- 前景/背景解耦的轻量多帧融合去雾（运动区域用时域减法，静止区用 DCP）
- 运动蒙版增强（增强增量仅作用于亮度通道的检测运动区域）
- 帧级退化检测 + 自动模块路由（无需用户选择算法）
- **零机器学习依赖**（纯 OpenCV + NumPy 传统图像处理）

---

## 🏗️ 算法管线

```
输入帧 F(t)
   │
   ▼
┌──────────────────────────────┐
│ 退化检测器                   │
│ → 雾浓度/眩光/亮度 统计评分   │ ← 新增 M5
└──────┬───────────────────────┘
       │ (根据退化类型自动路由)
       │
       ├── 🌫️ 去雾 ────────────────────┐
       │   Phase1: 时域雾层估计 (M1)    │
       │   Phase2: DCP 去雾 (M1)       │
       │   Phase3: 自适应融合 (M2)      │
       │   └── Phase4: 亮度增强 (M2)    │
       │
       ├── ☀️ 眩光抑制 (M3) ─── 新增 ───┤
       │   亮通道检测 → 局部压低 → 时域插值 │
       │
       ├── 🌙 低光照增强 (M4) ─ 新增 ───┤
       │   Retinex 分解 → 伽马 → CLAHE  │
       │
       └── 🌪️ 混合退化 ─── 编排器调度 ──┘
                      │
              增强帧输出
```

### 各模块详情

| 模块 | 方法 | 文件 |
|------|------|------|
| **M1** 时域雾层估计 | 运行式最小值跟踪，O(1) 每像素 | `algorithms/frame_diff.py` |
| **M1** 暗通道先验去雾 | DCP [He et al. 2009] + 引导滤波/快速模式 | `algorithms/dehaze.py` |
| **M2** 自适应融合 | sigmoid 运动幅值融合 temporal/DCP | `algorithms/enhance.py` |
| **M2** 亮度增强 | Lab-L 通道 CLAHE + Unsharp + 自适应伽马 | `algorithms/enhance.py` |
| **M3** 眩光抑制 🔥 | 亮通道检测 + 连通区域分析 + 亮度压低 | `algorithms/glare.py` |
| **M4** 低光增强 🔥 | Retinex 分解 + 照明层伽马 + 噪声感知 | `algorithms/low_light.py` |
| **M5** 退化检测 🔥 | 中段漂白率 + 垂直对比度衰减 + 暗像素统计 | `algorithms/degradation_detector.py` |
| **M6** 管线编排 🔥 | 可插拔配置 + auto_detect 路由 | `services/processing_service.py` |

> 🔥 = 本课题新增模块

---

## 📊 效果概览

在合成测试集上的定量评估（clean → degraded → enhanced，PSNR/SSIM）：

| 场景 | PSNR ↑ | SSIM ↑ |
|------|--------|--------|
| 🌫️ 雾霾 | 17.27 dB | 0.777 |
| ☀️ 眩光 | 20.39 dB | 0.785 |
| 🌙 低光照 | 9.96 dB | 0.677 |
| 🌪️ 混合退化 | 13.41 dB | 0.770 |

> 低光 PSNR 较低是正常现象：将暗帧从均值≈15 提亮到≈80 大幅改变了像素绝对值，但结构相似度 SSIM=0.68 说明纹理被很好地保留。

---

## 🚀 快速开始

### 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | ≥ 3.11 | 后端运行环境 |
| Node.js | ≥ 18 | 前端构建环境 |
| npm | ≥ 9 | 前端包管理 |

### 后端启动

```bash
cd backend

# 创建虚拟环境（推荐）
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py
# → http://localhost:8000
# → API 文档: http://localhost:8000/docs
```

### 前端启动

```bash
cd frontend

npm install
npm run dev
# → http://localhost:5173
```

### 使用流程

1. 浏览器打开 `http://localhost:5173`
2. 拖拽或点击上传行车视频（.mp4 / .avi / .mov，最大 500MB）
3. 调节算法参数（或保持默认）
4. 点击「开始处理」
5. 等待处理完成 → 对比预览 → 下载增强结果

---

## 📁 项目结构

```
fog-drive-enhancer/
├── frontend/                          # React 前端
│   └── src/
│       ├── App.tsx                    # 主布局（上传→处理→预览→导出）
│       ├── components/
│       │   ├── VideoUploader.tsx      # 拖拽上传
│       │   ├── VideoPlayer.tsx        # 原片/增强对比播放
│       │   ├── ControlPanel.tsx       # 算法参数调节
│       │   └── VideoExport.tsx        # 下载/清理
│       ├── api/index.ts               # FastAPI 通信层
│       └── types/index.ts             # 类型定义
├── backend/                           # Python 后端
│   ├── main.py                        # FastAPI 入口 + REST 路由
│   ├── requirements.txt               # 依赖清单
│   ├── algorithms/
│   │   ├── frame_diff.py              # M1: 时域雾层估计
│   │   ├── dehaze.py                  # M1: DCP 去雾
│   │   ├── enhance.py                 # M2: 融合 + 增强
│   │   ├── degradation_detector.py    # M5: 退化检测器 🔥
│   │   ├── glare.py                   # M3: 眩光抑制 🔥
│   │   └── low_light.py               # M4: 低光增强 🔥
│   ├── services/
│   │   ├── video_service.py           # 视频 I/O 封装
│   │   └── processing_service.py      # M6: 管线编排 🔥
│   └── scripts/
│       ├── evaluate.py                # M7: 定量评估框架
│       ├── baselines.py               # M8: 基线对比脚本
│       └── ablation.py                # M9: 消融实验框架
├── docs/
│   ├── PROJECT.md                     # 项目说明
│   ├── PRD.md                         # 技术 PRD
│   ├── product/                       # 产品设计文档
│   └── delivery/logs/                 # 开发日志
└── ops/
    └── RUNBOOK.md                     # 运维手册
```

---

## 🔌 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 健康检查 |
| `POST` | `/api/upload` | 上传视频 |
| `POST` | `/api/process/{id}` | 开始处理（可传 config） |
| `POST` | `/api/process/{id}/cancel` | 取消处理 |
| `GET` | `/api/status/{id}` | 查询进度 |
| `GET` | `/api/download/{id}` | 下载结果 |
| `GET` | `/api/preview/{id}` | 预览图 |
| `DELETE` | `/api/cleanup/{id}` | 清理文件 |

**处理参数（POST /api/process/{id} 的 JSON body）：**

```json
{
  "config": {
    "enable_fog": true,
    "enable_glare": true,
    "enable_low_light": true,
    "auto_detect": true,
    "omega": 0.95,
    "clahe_clip": 2.0
  }
}
```

---

## 📈 评估与消融

### 运行基线对比

```bash
# 对比 4 种方法：DCP-only / CLAHE-only / Retinex / DriveClear
python backend/scripts/baselines.py input.mp4 eval_results/ [original.mp4]
```

### 运行消融实验

```bash
# 8 种配置逐一 disable 模块，量化各模块贡献
python backend/scripts/ablation.py input.mp4 ablation_results/ [original.mp4]
```

消融配置包括：`full`、`no_fog`、`no_glare`、`no_ll`、`no_fusion`、`no_enhance`、`dcp_only`、`none`

### 单次评估

```bash
# PSNR / SSIM / MSE / RMSE
python backend/scripts/evaluate.py original.mp4 enhanced.mp4 report.json
```

---

## 🧰 技术栈

| 层 | 技术 | 用途 |
|----|------|------|
| 前端框架 | React 18 + Vite + TypeScript | UI |
| UI 组件 | Ant Design 5.x | 交互控件 |
| HTTP 通信 | Axios + FastAPI | 前后端通信 |
| 视频处理 | OpenCV 4.9 (cv2) | 帧读写、滤波、色彩空间 |
| 数值计算 | NumPy 1.26 | 矩阵运算、统计 |
| 图像评估 | scikit-image 0.23 | PSNR / SSIM |
| 后端服务器 | Uvicorn | ASGI 服务 |

**约束：** 本项目为数字图像处理课程项目，不使用任何机器学习/深度学习方法。

---

## 📅 时间线

| 阶段 | 任务 | 状态 |
|------|------|------|
| W1 (6/23-6/29) | 文献调研，DCP 去雾基线，全栈脚手架 | ✅ |
| W2 (6/30-7/6) | 眩光去除，低光增强，退化检测器 | ✅ |
| **W3 (7/7-7/13)** | **管线集成，基线对比，消融实验** | **进行中** |
| W4 (7/14-7/20) | 定量评估，报告撰写，PPT 准备 | 🔜 |

---

## 📄 许可证

本项目仅用作 **ECE4512 数字图像处理** 课程 Final Project 的学术展示。
