# RAG-ChromaDB

基于 **ChromaDB** 与 **阿里云百炼（DashScope）通义千问大模型** 构建的 RAG（检索增强生成）问答系统。

系统将本地文档切分、向量化后存入 ChromaDB 向量数据库；问答时先检索最相关的文档片段，再交给大模型生成基于知识库的回答，有效避免模型“凭空编造”内容。

---

## 功能特性

- 📄 **文档加载与切分**：基于 `RecursiveCharacterTextSplitter` 实现按字符智能切分，支持配置块大小与重叠窗口。
- 🔢 **批量向量化**：调用阿里云百炼 Embedding 接口批量生成向量，内置批处理控制，避免 API 限流。
- 🗄️ **持久化向量检索**：ChromaDB 本地持久化存储，支持 L2 距离快速相似度检索。
- 🤖 **RAG 问答链**：精准注入 Context 上下文，增强模型事实一致性。
- 🖥️ **交互式 CLI**：命令行交互式问答界面，开箱即用。

---

## 技术栈

| 模块 | 技术选型 | 版本/模型说明                          |
| :--- | :--- |:---------------------------------|
| **Vector DB** | ChromaDB | 本地持久化向量数据库                       |
| **Embedding** | DashScope Embeddings | `text-embedding-v3`              |
| **LLM** | 通义千问 Qwen / DeepSeek | `qwen-plus` / `deepseek-chat`    |
| **Splitter** | LangChain Text Splitters | `RecursiveCharacterTextSplitter` |
| **Runtime** | Python | 3.11.9                           |

---

## 项目结构

```
.
├── main.py                # 程序入口（数据入库与 CLI 交互调度）
├── config.py              # 全局配置中心（路径、API Key、超参数）
├── models.py              # LLM 与 Embedding 客户端工厂
├── document_loader.py     # 本地文本加载与切分逻辑
├── embedding_client.py    # 向量生成与批处理封装
├── vector_store.py        # ChromaDB 集合管理与 Top-K 检索
├── llm_client.py          # Prompt 组装与大模型交互
├── RAGPipeline.py         # 核心管道（Ingest / Query / Chat 流水线）
├── DATA/
│   ├── deepseek百度百科.txt # 知识库原始语料
│   └── CHROMA_DATA/       # ChromaDB 持久化数据目录（运行时自动生成）
├── .env                   # 环境变量配置文件（敏感凭证）
└── requirements.txt       # 项目依赖清单
```
---

## 分支说明

仓库目前维护两个并行分支，对应代码演进的两个阶段：

| 分支 | 说明 | 实现方式 |
| :--- | :--- | :--- |
| `main` | **原始版本**（手写实现） | 手写 `open()` 文件读取、手写分批向量化、原生 chromadb API、手写 Prompt 拼装 |
| `feature/langchain-refactor` | **LangChain 重构版** | `TextLoader` 加载、`langchain_chroma.Chroma` 存储、`ChatPromptTemplate` + LCEL 链式问答 |


---

## 快速开始

### 1. 安装依赖

```bash
pip install chromadb langchain langchain-openai langchain-community langchain-text-splitters openai python-dotenv
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件并填入凭证：

```ini
# Deepseek
DEEPSEEK_API_KEY='sk-密钥'
DEEPSEEK_API_URL='https://api.deepseek.com'
DEEPSEEK_API_MODEL='deepseek-chat'

# 阿里云百炼
DASHSCOPE_API_KEY='sk-密钥'
DASHSCOPE_API_URL='https://llm-qwe2oxuia5qhvcmo.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
DASHSCOPE_API_MODEL='qwen3.7-plus-2026-05-26' # 2026.9.1到期

```

### 3. 准备知识库数据

将目标文本文件（这里是[deepseek百度百科.txt](DATA/deepseek%E7%99%BE%E5%BA%A6%E7%99%BE%E7%A7%91.txt)）放入 `DATA/` 目录，并确保 `config.py` 中的 `DATA_PATH` 路径正确：

```python
DATA_PATH = os.path.join(os.path.dirname(__file__), "DATA", "deepseek百度百科.txt")
```

### 4. 运行系统

```bash
python main.py
```

---

## 运行效果示例

```text
==================================================
              AI-RAG 知识库问答系统
==================================================
[1/3] 正在加载并切分文档... 完成 (共 18 个分块)
[2/3] 正在生成文本向量 (Batch Size: 10)... 完成
[3/3] 数据已成功持久化至 ChromaDB.

系统已就绪，请输入您的问题（输入 'q' 退出）：

请输入问题：DeepSeek 是什么？
AI 回答：根据知识库记录，DeepSeek（深度求索）是一家专注于通用人工智能（AGI）研发的公司，旗下开发了包括 DeepSeek-V3、DeepSeek-R1 等多款高性能开源大语言模型，在逻辑推理、代码生成和数学解题上表现优异。

请输入问题：DeepSeek-R1 采用了什么核心技术？
AI 回答：DeepSeek-R1 采用了大规模强化学习（RL）训练技术，在没有大量人工标注冷启动数据的情况下，通过后训练自主涌现出深度思考与自省推理能力。

请输入问题：q
系统已退出。
```

---

## 核心参数说明

| 配置项 | 环境变量名 | 默认值 | 作用与调优建议 |
| :--- | :--- | :--- | :--- |
| **分块大小** | `CHUNK_SIZE` | `150` | 单个文档切片的最大字符数。短文本提高精准度，长文本保留上下文连贯性。 |
| **重叠长度** | `CHUNK_OVERLAP` | `30` | 相邻文本块的重叠字符数。防止边界信息截断丢失语义。 |
| **向量化批次** | `BATCH_SIZE` | `10` | 批量请求 Embedding 接口的文本数量，避免触发 API 速率限制。 |
| **检索数量** | `TOP_K` | `5` | 每次问答从 ChromaDB 召回的最相似片段数量。 |
| **采样温度** | `TEMPERATURE` | `0.0` | 建议设为 `0` 以确保 RAG 输出严格遵循召回文档，减少幻觉。 |

---
## 常见问题排查 (Troubleshooting)

1. **uvloop 安装失败**

   问题：执行 `pip install -r requirements.txt` 时报错 `RuntimeError: uvloop does not support Windows`，导致依赖安装中断。

   原因：uvloop 只支持 Linux/macOS，是从其他平台导出的依赖文件带进来的。解决办法：从 requirements.txt 中删除 `uvloop==0.22.1` 这一行，再重新安装，不影响 Windows 上运行。

2. **缺少 python-dotenv 模块**

   问题：运行 `main.py` 报 `ModuleNotFoundError: No module named 'dotenv'`。

   原因：之前 uvloop 报错中断，导致后面的包没装全。解决办法：执行 `pip install python-dotenv`（注意包名是 python-dotenv，不是 dotenv），或重新完整安装 requirements。

3. **读取文件编码报错**

   问题：运行时出现 `UnicodeDecodeError: 'gbk' codec can't decode...`。

   原因：Windows 上 open() 默认用 GBK 编码，而 `deepseek百度百科.txt` 是 UTF-8 保存的，编码不匹配。解决办法：在 `document_loader.py` 中改为 `open(file_path, 'r', encoding='utf-8')`，显式指定编码。

4. **向量维度不匹配错误**

   问题：`Collection expecting embedding with dimension of X, got Y`
   原因：更换了 Embedding 模型（例如从 1536 维切换至 1024 维），但旧的 ChromaDB 持久化文件依然保留了原维度的集合定义。
   解决：清空持久化目录后重新运行程序重建索引

5. **vpn影响访问github**

   问题：无法访问github，fatal: unable to access 'https://github.com/renduermai/RAG-ChromaDB.git/': Failed to connect to github.com port 443 via 127.0.0.1 after 2117 ms: Could not connect to server

   原因：访问github需要打开vpn，并设置到全局模式。在pip 的时候要关闭vpn
---

## License

本项目仅供学习与技术研究使用。