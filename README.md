# Doc Diff Agent

Doc Diff Agent 是一个面向 Windows 桌面场景的文档比对与问答工具，聚焦于法规、制度、合同、规范类文档的版本管理、语义差异识别和检索式问答。

项目当前以 PySide6 桌面应用的形式运行，核心能力覆盖文档导入、版本比对、RAG 问答与差异报告导出。

## 核心能力

- 文档入库与版本管理：支持导入 PDF、DOCX 文档，自动计算文件哈希避免重复导入，并支持同一文档下新增版本。
- 语义级文档比对：先做章节对齐，再做段落级语义匹配，结合嵌入模型与大模型分类差异类型。
- 差异结果输出：可输出新增、删减、微调、实质修改、重写、格式变化等结果。
- 文档问答：基于本地切块、向量检索和大模型回答实现 RAG 问答，支持当前文档、基准版本、目标版本、标准库、全库等检索范围。
- 报告导出：支持导出 HTML 与 DOCX 对比报告，包含差异统计、风险等级、相似度和详细内容。
- 模型接入：当前可用 OpenAI 兼容接口，支持可选本地 embedding 模型路径；Azure Provider 已预留接口但尚未完成实现。

## 技术栈

- 桌面界面：PySide6
- 文档解析：Docling、PyMuPDF、python-docx
- 编排与状态流：LangGraph
- 向量检索：FAISS
- 数据持久化：SQLite + SQLAlchemy
- 模型能力：OpenAI Compatible API、sentence-transformers

## 运行要求

- Windows
- Python 3.11+

建议使用虚拟环境。

## 快速开始

### 1. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

### 2. 启动应用

```powershell
python main.py
```

应用启动后可在设置页配置模型提供方、API Key、本地 embedding 路径和数据目录。

## 配置说明

- 默认配置文件位置：%APPDATA%\DocDiffAgent\config.json
- 默认数据目录：%APPDATA%\DocDiffAgent

默认数据目录下会生成以下内容：

- docs/：导入后的原始文档副本
- parsed/：解析后的 DocumentIR JSON
- exports/：对比结果导出文件
- SQLite 数据文件与向量索引

## 文档处理说明

- PDF：standard 模式优先使用 Docling，失败后回退到 PyMuPDF；fast 模式直接使用 PyMuPDF。
- DOCX：使用 python-docx 提取文本与结构。

## 测试

```powershell
pytest
```

## 项目结构

```text
app/
  agent/      LangGraph 工作流
  config/     配置与密钥处理
  core/       领域模型、解析、比对、检索、模型适配
  db/         数据表与仓储层
  services/   导入、比对、问答、报告服务
  ui/         桌面界面
assets/         模板、字体、图标
tests/          自动化测试
```

## 当前边界

- 当前主运行形态为 Windows 桌面应用。
- 当前主要对接 OpenAI 兼容模型接口。
- Azure OpenAI Provider 尚未完成，不建议在当前版本中启用。

## 开发建议

- 修改文档解析链路后，优先运行 parser 与 services 相关测试。
- 修改比对逻辑后，优先运行 diff、agent、services 相关测试。
- 如果接入本地 embedding，请确保模型目录可被 sentence-transformers 正确加载。
