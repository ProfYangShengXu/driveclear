# fog-drive-enhancer 运维手册

## 环境要求

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| Python | ≥ 3.11 | 后端运行环境 |
| Node.js | ≥ 18 | 前端构建环境 |
| npm | ≥ 9 | 前端包管理 |

## 快速启动

### 后端

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
# 服务运行在 http://localhost:8000
# API 文档: http://localhost:8000/docs
```

### 前端

```bash
cd frontend

# 安装依赖
npm install

# 开发模式（热更新）
npm run dev
# 运行在 http://localhost:5173

# 生产构建
npm run build
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| POST | `/api/upload` | 上传视频 |
| POST | `/api/process/{video_id}` | 开始处理 |
| POST | `/api/process/{video_id}/cancel` | 取消处理 |
| GET | `/api/status/{video_id}` | 查询进度 |
| GET | `/api/download/{video_id}` | 下载结果 |
| GET | `/api/preview/{video_id}` | 预览图 |
| DELETE | `/api/cleanup/{video_id}` | 清理文件 |

## 交付前自检清单

### Agent 可执行项

- [ ] 后端依赖安装成功: `cd backend && pip install -r requirements.txt`
- [ ] 后端可启动: `cd backend && python -c "from algorithms import TemporalFogEstimator, dehaze, enhance_frame; print('算法模块加载成功')"`
- [ ] 前端依赖安装成功: `cd frontend && npm install`
- [ ] 前端可构建: `cd frontend && npm run build`
- [ ] 前端 lint 无报错: `cd frontend && npx tsc --noEmit`
- [ ] 后端 FastAPI 启动无报错: `cd backend && python -c "from main import app; print('FastAPI 加载成功')"`

### 用户需确认项

- [ ] 实际上传一段雾天行车视频，确认处理效果
- [ ] 检查导出视频能否正常播放
- [ ] 调整算法参数测试不同效果
