# fog-drive-enhancer

> 基于图像减法 + 图像增强方法，对大雾中开车的视频做逐帧去雾处理，增强障碍物细节。前端可导入/导出视频。

## 概述

本工具通过四阶段算法管线处理雾天行车视频：
1. **时域雾层估计 (Temporal Fog Layer Estimation)** — 利用滑动窗口中连续帧的低分位数估计雾层，帧间减法初步去雾
2. **暗通道先验去雾 (Dark Channel Prior)** — 经典 DCP 算法恢复场景辐射
3. **自适应融合 (Adaptive Fusion)** — 按运动幅值动态融合时域结果与 DCP 结果，运动区域突出障碍物
4. **图像增强 (Enhancement)** — CLAHE 局部对比度 + Unsharp Mask 锐化 + 自适应伽马校正

## 项目结构

```
fog-drive-enhancer/
├── frontend/                         # React 前端
│   └── src/
│       ├── App.tsx                   # 主应用（上传→处理→预览→导出布局）
│       ├── main.tsx                  # 入口
│       ├── components/
│       │   ├── VideoUploader.tsx     # 拖拽/点击上传视频
│       │   ├── VideoPlayer.tsx       # 原始/增强视频预览
│       │   ├── ControlPanel.tsx      # 算法参数调节
│       │   └── VideoExport.tsx       # 下载/清理
│       ├── api/index.ts              # FastAPI 通信层
│       └── types/index.ts            # 类型定义
├── backend/                          # Python 后端
│   ├── main.py                       # FastAPI 入口 + REST 路由
│   ├── requirements.txt              # Python 依赖
│   ├── algorithms/
│   │   ├── frame_diff.py             # Phase1: 时域雾层估计 + 帧间减法
│   │   ├── dehaze.py                 # Phase2: 暗通道先验去雾
│   │   └── enhance.py                # Phase3+4: 融合 + CLAHE + 锐化 + 伽马
│   └── services/
│       ├── video_service.py          # 视频 I/O 封装
│       └── processing_service.py     # 四阶段管线调度
├── docs/
│   ├── PROJECT.md                    # 本文件
│   └── delivery/logs/                # 阶段零更新日志
└── ops/
    └── RUNBOOK.md                    # 运维手册
```

## 算法管线

```
输入帧 F(t)
  │
  ▼
┌─────────────────────────────────────┐
│ Phase1: 时域雾层估计                  │
│ 缓冲区 [F(t-15)..F(t-1)]             │
│ → 像素级 10% 分位数 → 雾层 fog(t)    │
│ → D(t) = F(t) - fog(t)              │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│ Phase2: 暗通道先验 (DCP)             │
│ → 大气光 A + 传输率 t(x) → J(t)      │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│ Phase3: 自适应融合                    │
│ 运动检测 → α = sigmoid(motion-0.3)   │
│ Out = α·D(t) + (1-α)·J(t)           │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│ Phase4: 增强                         │
│ CLAHE(Lab-L) → Unsharp Mask → γ 校正 │
└─────────────┬───────────────────────┘
              │
        输出增强帧
```

## 快速开始

### 后端

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
python main.py
# → http://localhost:8000
# → API 文档: http://localhost:8000/docs
```

### 前端

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| POST | `/api/upload` | 上传视频 |
| POST | `/api/process/{id}` | 开始处理 |
| POST | `/api/process/{id}/cancel` | 取消处理 |
| GET | `/api/status/{id}` | 查询进度 |
| GET | `/api/download/{id}` | 下载结果 |
| GET | `/api/preview/{id}` | 预览图 |
| DELETE | `/api/cleanup/{id}` | 清理文件 |

## 变更索引

- [2026-06-18 项目初始化搭建](delivery/logs/2026-06-18-init-project.md) — 全栈脚手架 + 算法实现

## 技术栈

| 层 | 技术 |
|----|------|
| 前端 | React 18 + Vite + TypeScript + Ant Design |
| 后端 | Python 3.11+ + FastAPI |
| 算法 | OpenCV + NumPy + scikit-image |
| 通信 | REST API |
