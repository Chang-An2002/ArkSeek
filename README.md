# ArkSeek

ArkSeek 是本机运行的 QQ 聊天机器人与只读管理后台。它结合分层记忆、本地混合检索、PRTS Wiki 与联网搜索，为回答提供资料依据。

## 当前功能

- QQ 私聊与群聊接入，使用官方 Python SDK。
- 最近对话与摘要、个人画像、历史问答检索和本地知识库四层记忆。
- BM25、向量召回、RRF 融合、精排与可信度判断。
- 复合问题拆分、依赖调度和并行查询。
- 管理后台：会话与用户浏览、画像来源和保护状态、请求状态、排队及分步计时、运行日志。

本仓库提供公开源码和虚构示例，不包含开发者的研究资料、真实聊天、画像库、密钥、运行日志或内部 Git 历史。模型、向量索引与 Python 环境需要在使用者的电脑上建立。

## 先体验管理后台

后台演示只需要 Python 与 aiohttp，不加载模型或调用 LLM。以下命令在 Windows PowerShell 中执行：

```powershell
Set-Location arkseek
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install aiohttp==3.14.3
& .\.venv\Scripts\python.exe -m admin --demo
```

打开 `http://127.0.0.1:8766`。演示使用临时虚构数据，退出后清理。也可以运行 `./Start-Admin.ps1 -Demo` 启动后台并打开浏览器；此方式创建独立后台进程，启动输出会给出停止命令。

已有自己的 ArkSeek 数据时使用 `--live` 或 `Start-Admin.ps1 -Live` 只读查看。端口占用时使用 `--port 8767` / `-Port 8767`。后台与 QQ 机器人各自运行；关闭后台不会关闭机器人。

## 运行机器人

完整机器人有模型和文档处理依赖。本机开发环境为 Windows、Python 3.14；精排可使用 Intel OpenVINO。锁定依赖记录了开发环境，其他系统或硬件的完整安装仍需按实际环境验证。

```powershell
Set-Location arkseek
& .\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
# 没有 .env 时才从示例创建，不覆盖已有配置。
Copy-Item .env.example .env
```

在 `.env` 填写自己的 QQ 应用和模型服务配置。配置示例中部分路径和知识库来源需要按自己的环境调整。QQ 应用注册与服务费用由使用者自行管理。

仓库内 `data/knowledge_base/demo.txt` 是虚构技术示例，可用于理解本地检索。将自己的资料副本放入知识库目录；外部来源及本机配置参考 `src/kb_sources.py`。请确认自己拥有资料使用权限。

```powershell
Set-Location src
& ..\.venv\Scripts\python.exe qq_bot.py
# 命令行对话入口
& ..\.venv\Scripts\python.exe chat_cli.py
```

机器人或 CLI 会调用已配置的真实服务，并保存自己的会话和画像。后台演示命令不会启动机器人。

## 存储和身份

SQLite 保存原始会话与画像，Chroma 是可重建的派生索引。群聊按群保存对话与摘要，个人画像按群成员身份保存；同昵称、不同群和私聊身份不自动合并。

管理后台使用只读 SQLite 连接，读取存储值、来源与保护规则状态，不初始化会迁移结构的 MemoryStore。新请求监测由新版机器人启动后产生，旧请求缺失的计时不补造。进程心跳不等于 QQ 实时连接状态。

## 验证与边界

- 只读后台与 QQ 发送适配的临时虚构数据测试：23 项通过。
- 匿名群聊结构回归：21 项通过。
- 管理页已在桌面和窄屏浏览器检查，会话/画像跳转、失败筛选、计时、日志搜索与主题切换通过。

```powershell
# 在 arkseek 目录：后台测试仅需 aiohttp，使用临时数据和假发送接口。
& .\.venv\Scripts\python.exe -m unittest admin.test_admin -q
```

这些结果不代表任意真实 QQ 场景、所有硬件或模型服务均已验证。源代码保留了开发过程中使用的模型/检索实验脚本；依赖私人语料的旧实验不应在此示例知识库上宣称复现原结果。默认运行 `chat_cli.py` 或部分旧演示会触碰本机真实数据，请先检查其数据路径。

## 后续路线

第一版后台提供只读监测与浏览。画像纠错、记忆停用、操作历史、机器人启停与配置、知识库维护和更完整的连接/来源观测均为后续计划，尚未实现。

管理后台说明见 [docs/admin-console.md](docs/admin-console.md)。
