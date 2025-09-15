# 抄袭检测可视化前端

基于 React + TypeScript + Tailwind CSS 构建的抄袭检测可视化界面，支持左右对照展示文档相似度分析结果。

## 功能特色

### 🔍 智能检测
- **三种检测模式**：快速、详细、全面检测
- **自定义阈值**：支持 50%-95% 相似度阈值设置
- **实时进度**：检测过程可视化展示

### 🎨 可视化高亮
- **多层级高亮**：
  - 🔴 红色 - 高相似度 (80%+)
  - 🟠 橙色 - 中相似度 (60-80%)
  - 🟡 黄色 - 低相似度 (50-60%)
- **悬浮提示**：显示具体相似度分值和来源信息
- **点击定位**：点击高亮区域自动滚动到对应位置

### ⚡ 同步交互
- **双向滚动同步**：左右面板联动滚动
- **精确位置映射**：相似内容精准对照
- **流畅动画**：平滑的交互体验

### 📁 文件支持
- **多格式支持**：TXT、MD、DOC、DOCX 文件
- **拖拽上传**：便捷的文件操作方式
- **大文件处理**：支持长文本检测

## 快速开始

### 1. 启动后端服务

确保抄袭检测 API 服务正在运行：

```bash
# 在项目根目录
python run.py
# 或者
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API 服务将在 http://localhost:8000 启动

### 2. 启动前端开发服务器

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

前端界面将在 http://localhost:5173 启动

### 3. 开始使用

1. **输入或上传文档**：在左侧面板输入或上传待检测的文本
2. **配置检测参数**：选择检测模式和相似度阈值
3. **开始检测**：点击"开始检测"按钮
4. **查看结果**：右侧面板显示相似度分析结果
5. **交互对照**：点击高亮区域查看详细信息

## 项目结构

```
frontend/
├── src/
│   ├── components/
│   │   ├── DocumentComparer.tsx      # 主对照组件
│   │   └── SimilarityHighlighter.tsx # 高亮渲染组件
│   ├── hooks/
│   │   ├── usePlagiarismDetection.ts # 检测逻辑Hook
│   │   └── useSyncScrolling.ts       # 同步滚动Hook
│   ├── api/
│   │   └── plagiarismApi.ts          # API 客户端
│   └── index.css                     # 样式文件
├── package.json
└── vite.config.ts
```

## 技术栈

- **前端框架**：React 18 + TypeScript
- **构建工具**：Vite
- **样式方案**：Tailwind CSS + 自定义CSS
- **HTTP 客户端**：Axios
- **图标库**：Lucide React

## API 接口

### 检测端点

`POST /api/v1/detection/check`

请求体：
```json
{
  "content": "待检测文本内容",
  "mode": "detailed",
  "threshold": 0.75
}
```

响应：
```json
{
  "task_id": "检测任务ID",
  "status": "completed",
  "total_matches": 5,
  "paragraph_matches": [...],
  "sentence_matches": [...],
  "processing_time": 2.34,
  "created_at": "2024-03-15T10:30:00Z"
}
```

### 结果查询

`GET /api/v1/detection/{task_id}`

返回相同格式的检测结果

## 开发指南

### 本地开发

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 类型检查
npm run build
```

### 部署构建

```bash
# 构建生产版本
npm run build

# 预览构建结果
npm run preview
```

构建产物将生成在 `dist/` 目录中。

## 浏览器支持

- Chrome/Edge 90+
- Firefox 90+
- Safari 14+

## Linus 原则实践

遵循 **"代码胜于雄辞"** 的开发哲学：

- ✅ **简洁实用**：专注核心功能，避免过度设计
- ✅ **类型安全**：全 TypeScript 开发，减少运行时错误
- ✅ **性能优先**：组件懒加载，虚拟滚动优化
- ✅ **用户体验**：流畅交互，清晰反馈

## 许可证

MIT License