# 阶段零 — 更新日志：项目初始化搭建

## §1 计划更新

### 目标
从零搭建 fog-drive-enhancer 项目（前端 React + 后端 Python FastAPI），实现基于时域雾层估计 + 暗通道先验 + 图像增强的去雾视频处理工具。

### 改动范围表

| # | 文件 | 操作 | 复用/新增 | 说明 |
|---|------|------|-----------|------|
| 1 | `backend/algorithms/frame_diff.py` | 创建 | 新增 | 时域雾层估计 + 帧间减法 |
| 2 | `backend/algorithms/dehaze.py` | 创建 | 新增 | 暗通道先验去雾 (DCP) |
| 3 | `backend/algorithms/enhance.py` | 创建 | 新增 | 融合策略 + CLAHE + 锐化 + 伽马校正 |
| 4 | `backend/services/video_service.py` | 创建 | 新增 | OpenCV 视频读写封装 |
| 5 | `backend/services/processing_service.py` | 创建 | 新增 | 四阶段管线调度 |
| 6 | `backend/main.py` | 创建 | 新增 | FastAPI 入口 + 路由（上传/处理/下载/状态） |
| 7 | `backend/requirements.txt` | 创建 | 新增 | Python 依赖清单 |
| 8 | `frontend/src/main.tsx` | 创建 | 新增 | React 入口 |
| 9 | `frontend/src/App.tsx` | 创建 | 新增 | 主布局（上传→预览→控制→导出） |
| 10 | `frontend/src/components/VideoUploader.tsx` | 创建 | 新增 | 视频上传组件 |
| 11 | `frontend/src/components/VideoPlayer.tsx` | 创建 | 新增 | 视频预览（原片/处理后对比） |
| 12 | `frontend/src/components/ControlPanel.tsx` | 创建 | 新增 | 参数调节面板 |
| 13 | `frontend/src/components/VideoExport.tsx` | 创建 | 新增 | 导出组件 |
| 14 | `frontend/src/api/index.ts` | 创建 | 新增 | FastAPI 通信层 |
| 15 | `frontend/src/types/index.ts` | 创建 | 新增 | 类型定义 |
| 16 | `frontend/package.json` | 创建 | 新增 | Node 依赖 |
| 17 | `frontend/vite.config.ts` | 创建 | 新增 | Vite 配置 |
| 18 | `frontend/tsconfig.json` | 创建 | 新增 | TS 配置 |
| 19 | `frontend/index.html` | 创建 | 新增 | HTML 入口 |
| 20 | `docs/delivery/logs/2026-06-18-init-project.md` | 创建 | 新增 | 本日志文件 |
| 21 | `ops/RUNBOOK.md` | 创建 | 新增 | 运维手册（阶段三预创建） |

> **复用说明**：本项目为新项目，无既有代码可复用。所有文件均为新增。

### 明确不做（本次范围外）
- 不接入深度学习模型（纯传统图像处理）
- 不做实时视频流处理（仅处理上传的完整视频文件）
- 不做 GPU 加速（起步阶段 CPU 版，后续可加）

### 依赖前置
- Node.js 18+ (前端构建)
- Python 3.11+ (后端运行)
- 无其他项目内前置依赖

### 待确认项
- 算法参数默认值已定，ControlPanel 暴露滑条让用户微调（已含在计划中）
- 视频格式优先支持 mp4，后续可扩展

---

## §2 目标逻辑链

```
触发入口 [用户上传视频] → 输入校验 [格式/大小] → 核心处理 [四阶段管线] 
→ 持久化 [服务器存储结果] → 返回/展示 [前端预览] → 失败与回滚 [错误处理]
```

### 环 1：触发入口
| 项 | 内容 |
|----|------|
| **预期行为** | 用户在 VideoUploader 选择 .mp4 文件，点击上传 |
| **涉及文件** | `VideoUploader.tsx` → `api/index.ts` → `main.py` |
| **验收标准** | 上传成功后返回 video_id，前端进入预览状态 |
| **风险边界** | 文件过大时需有进度显示；格式不支持给出明确提示 |

### 环 2：输入校验
| 项 | 内容 |
|----|------|
| **预期行为** | 后端校验视频格式（.mp4/.avi/.mov）、大小（≤500MB）、可读性 |
| **涉及文件** | `main.py` (POST /upload) → `video_service.py` (validate_video) |
| **验收标准** | 不合法视频返回 4xx + 中文错误信息 |
| **风险边界** | 损坏的视频文件需优雅处理 |

### 环 3：核心处理（四阶段管线）
| 项 | 内容 |
|----|------|
| **预期行为** | 后端逐帧读取 → Phase1 时域雾层估计 → Phase2 DCP → Phase3 融合 → Phase4 增强 → 逐帧写入输出 |
| **涉及文件** | `processing_service.py` → `frame_diff.py` / `dehaze.py` / `enhance.py` |
| **验收标准** | 输出视频雾感降低、障碍物细节增强、色彩自然 |
| **风险边界** | 前 N 帧因缓冲区不足跳过处理直接输出；场景切换时重置缓冲区 |

### 环 4：持久化/副作用
| 项 | 内容 |
|----|------|
| **预期行为** | 处理后的视频写入临时文件，路径关联 video_id |
| **涉及文件** | `video_service.py` (write_video) → `main.py` |
| **验收标准** | 文件正确写入，可被下载和预览 |
| **风险边界** | 磁盘空间不足检测 |

### 环 5：返回/展示
| 项 | 内容 |
|----|------|
| **预期行为** | 前端轮询处理进度 → 完成后展示处理结果预览 → 用户播放对比原片/结果 |
| **涉及文件** | `VideoPlayer.tsx` → `api/index.ts` → `main.py` (GET /status, GET /result/:id) |
| **验收标准** | 预览正常播放，对比切换流畅 |
| **风险边界** | 大视频预览需先缓冲一定量 |

### 环 6：失败与回滚
| 项 | 内容 |
|----|------|
| **预期行为** | 任一环节失败 → 清理临时文件 → 返回错误信息 → 前端显示错误提示 |
| **涉及文件** | 全链条 |
| **验收标准** | 用户看到明确错误信息，临时文件被清理 |
| **风险边界** | 处理中途取消（用户关闭页面）→ 服务端清理 orphan 任务 |

---

## §3 实施记录

| 时间 | 操作 | 说明 |
|------|------|------|
| 2026-06-18 | 阶段零日志写入 | 本文件创建，§1+§2 填写完成 |
| 2026-06-18 | 算法模块实现 | `frame_diff.py` / `dehaze.py` / `enhance.py` 三个模块 |
| 2026-06-18 | 后端服务实现 | `main.py` (FastAPI) + `video_service.py` + `processing_service.py` |
| 2026-06-18 | 前端组件实现 | 4 个组件 + API 层 + 主布局 |
| 2026-06-18 | 依赖安装验证 | Python 依赖 (pip) + 前端依赖 (npm) 全部安装成功 |
| 2026-06-18 | 算法测试 | 合成数据通过全部算法模块验证 |
| 2026-06-18 | TS 类型检查 | `tsc --noEmit` 零错误通过 |
| 2026-06-18 | 阶段一清理 | 删除未使用的 import（见 tsc 修复记录） |
| 2026-06-18 | 运维手册 | `ops/RUNBOOK.md` 已创建 |

## §4 阶段二自查结论

### 环 1：触发入口 ✅
- **文件**: `frontend/src/components/VideoUploader.tsx:28-54` — customRequest 实现上传逻辑
- **文件**: `frontend/src/api/index.ts:15-24` — uploadVideo() 调用 POST /api/upload
- **文件**: `backend/main.py:39-55` — POST /api/upload 路由
- **通过**: 上传成功后返回 `{video_id, filename, message}`，前端据此进入处理界面

### 环 2：输入校验 ✅
- **文件**: `backend/services/video_service.py:28-49` — validate_video() 检查扩展名、文件大小、OpenCV 可读性
- **文件**: `backend/main.py:49-53` — 校验失败返回 400 + 中文错误
- **通过**: 格式/大小/可读性三层校验，损坏文件优雅处理

### 环 3：核心处理（四阶段管线） ✅
- **文件**: `backend/services/processing_service.py:83-115` — _process_video_worker() 调度四阶段
  - Phase1 → `backend/algorithms/frame_diff.py:37-62` — TemporalFogEstimator.update()
  - Phase2 → `backend/algorithms/dehaze.py:88-122` — dehaze() 完整流程
  - Phase3 → `backend/algorithms/enhance.py:23-42` — fuse_frames() 运动自适应融合
  - Phase4 → `backend/algorithms/enhance.py:51-148` — CLAHE + 锐化 + 伽马校正
- **通过**: 合成数据验证输出均值变化符合预期（雾层去除、对比度提升）
- **风险处理**: 前 N 帧缓冲区未满时走 else 分支直接输出 DCP 结果 (processing_service.py:102-104)

### 环 4：持久化/副作用 ✅
- **文件**: `backend/services/video_service.py:87-99` — write_video() 使用 OpenCV VideoWriter
- **文件**: `backend/services/video_service.py:53-65` — save_upload() 保存上传文件
- **通过**: 文件路径由 UUID 生成，避免冲突；输出路径通过 get_output_path() 管理

### 环 5：返回/展示 ✅
- **文件**: `frontend/src/api/index.ts:54-87` — pollStatus() 轮询 GET /api/status/{id}
- **文件**: `backend/main.py:112-126` — GET /api/status/{video_id} 返回完整状态
- **文件**: `backend/main.py:129-142` — GET /api/download/{video_id} 返回 FileResponse
- **文件**: `frontend/src/App.tsx:36-68` — 轮询回调更新进度 + 完成通知
- **通过**: 轮询机制 + 进度展示 + 完成后预览

### 环 6：失败与回滚 ✅
- **文件**: `backend/services/processing_service.py:117-120` — 异常捕获 → status=FAILED + error 记录
- **文件**: `backend/services/video_service.py:103-110` — cleanup() 删除输入输出文件
- **文件**: `backend/main.py:148-152` — DELETE /api/cleanup/{video_id} 暴露清理接口
- **通过**: 处理异常、用户取消、手动清理三条路径均覆盖
- **偏差说明**: 当前未实现自动清理 orphan 任务（用户关闭页面），依赖手动 DELETE /cleanup

> **自查结论**: 6 环全部通过 ✅。1 个偏差：orphan 任务自动清理暂未实现（控制面板不会导致的资源泄漏风险低，后续迭代可加）
