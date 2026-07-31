# 提交说明与运行指南

本文档用于 Agent Memory Leaderboard 的提交说明。它说明本仓库如何
运行、Docker 部署方式、Add/Search HTTP 封装位置、论文来源，以及这个
独立实现相对于论文公开规格完成的工程工作。

## 项目性质与论文来源

本仓库是对以下公开技术报告的 clean-room（独立重建）实现：

- 论文：**Mi-Memory: A Lifecycle Memory Framework for Personal AI**
- arXiv：[2607.18975](https://arxiv.org/abs/2607.18975)
- 项目主页：[Darwin Agent Team / Mi-Memory](https://darwin-agent.github.io/Mi-Memory/)
- 报告署名：**Darwin Agent Team**

技术报告第 34 页列出的核心贡献者为 Xule Liu、Hanlin Teng、Chao Li、
Yanan Ni、Kun Shao 与 Jian Luan；其中 Xule Liu、Hanlin Teng、Chao Li
为共同第一作者，Kun Shao 与 Jian Luan 为通讯作者。完整贡献者名单以
原报告为准。

本仓库不是论文作者发布的原始实现。它仅依据论文、附录和公开数据格式
完成独立工程重建；不包含或声称拥有未公开的原始提示词、内部
MemFuseBench、私有部署端点或论文作者的源代码。

## 本地运行

要求：Python 3.11 或更新版本。先安装项目：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

仅运行本地、无模型调用的基础服务：

```bash
mimemory --root .mimemory ingest "My keys are in the kitchen." --source-id turn-1
mimemory --root .mimemory recall "Where are my keys?"
```

严格论文运行时使用远程模型角色。凭据只保存在本地 `.env`，该文件被 Git
忽略，绝不能提交。每次真实调用必须显式授权：

```bash
MIMEMORY_LIVE_PROVIDER_APPROVED=1 \
mimemory-leaderboard --runtime paper --root .mimemory-api \
  --host 127.0.0.1 --port 8765
```

如果要让局域网设备访问，将 `--host 127.0.0.1` 改为 `--host 0.0.0.0`。
生产或公网部署应通过反向代理提供 HTTPS 和访问控制，而不是直接暴露本地
端口。

## Docker 启动命令

构建和启动：

```bash
docker build -t mi-memory-add-search .
docker run --rm \
  -p 8765:8765 \
  --env-file .env \
  -e MIMEMORY_LIVE_PROVIDER_APPROVED=1 \
  -v mimemory-data:/data \
  mi-memory-add-search
```

或使用 Compose：

```bash
MIMEMORY_LIVE_PROVIDER_APPROVED=1 docker compose up --build
```

容器监听 `8765`，健康检查为 `GET /health`。记忆数据保存于 `/data`
挂载卷；不将 `.env`、评测数据、请求正文或模型密钥写入镜像或 Git。

## Add/Search API 封装

HTTP 处理器位于 `src/mimemory/leaderboard.py`，CLI 入口为
`mimemory-leaderboard`。严格论文模式由 `PaperLeaderboardAdapter` 调用
`MemStackRuntime`；默认 `paper` 运行时不会在服务启动时发起模型调用，
只有实际 Add 或 Search 请求才会调用已授权的模型端点。

| 方法 | 主路径 | 兼容路径 | 作用 |
| --- | --- | --- | --- |
| `POST` | `/add` | `/v1/add` | 同步写入一个对话或 session |
| `POST` | `/search` | `/v1/search` | 在指定用户的记忆范围内检索 |
| `GET` | `/health` | - | 服务健康检查 |

### Add

请求体：

```json
{
  "request_id": "session-001",
  "user_id": "user-001",
  "session_id": "session-001",
  "messages": [
    {
      "role": "user",
      "timestamp": 1684549260000,
      "content": "The training bag is in the car."
    }
  ]
}
```

成功响应：

```json
{
  "success": true,
  "request_id": "session-001",
  "user_id": "user-001",
  "session_id": "session-001"
}
```

Add 在返回前完成写入。每个 `user_id` 使用独立的哈希存储目录；同一
`request_id` 携带完全相同内容时是幂等的，若复用 ID 但正文不同则返回
`HTTP 400`。

### Search

请求体：

```json
{
  "user_id": "user-001",
  "query": "Where is the training bag?",
  "top_k": 12,
  "options": []
}
```

成功响应：

```json
{
  "data": [
    {
      "id": "memory-id",
      "content": "The training bag is in the car.",
      "score": 0.98,
      "created_at": "2026-07-30T00:00:00+00:00"
    }
  ]
}
```

Search 支持多选题 `options`，并返回按相关性降序排列的结果。用户隔离是
硬边界，未知用户或无匹配内容返回 `{ "data": [] }`。

完整平台协议、鉴权和数据保留说明见 [SUBMISSION.md](SUBMISSION.md)。

## 实现改动概览

相对于论文公开描述，本仓库实现了以下可执行模块：

1. **MemStack**：L0/L1/L2/SM 分层记忆、事实抽取、向量/BM25/子查询三路
   检索、加权 RRF、独立重排、受保护上下文打包和诊断 trace。
2. **MemSense**：五个独立且可审计的 IKB 构建步骤、类别/session/日期索引、
   VR/VS/TTL 的 IKB-first 路由、最多八张图像的残余视觉检索接口。
3. **MemFuse**：三临时区 FusionSession、原子设备事件保留、持久化
   atomic-event/MemoryPack 双层图，以及 `BELONG`、`CAUSES` 因果边。
4. **D2ACCI 与 E2MEND**：六阶段对齐诊断、类别根因汇总、版本化策略、
   prompt 完整性门、Critic、UCB1、pending champion、回滚和审计记录。
5. **LiteMem**：Markdown frontmatter、style/profile 单例、日记写前日志、
   `SUMMARY_END` 标记、Eq. 20/21 评分项、惰性日记行窗口读取、纠错文件、
   原子索引和可选 Git 审计。
6. **评测与服务**：LoCoMo、PersonaMem-V2、LongMemEval 适配层；LongMemEval-S
   session 级 Add/Search 断点续跑脚本；以及 Leaderboard 兼容 HTTP 服务。

每项公开规格与实现文件的对应关系见
[PAPER_IMPLEMENTATION_CHECKLIST.md](PAPER_IMPLEMENTATION_CHECKLIST.md)。

## 验证与边界

离线测试：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -v
```

本实现不声称复现论文的实验分数。只有在单独批准、固定模型和评测配置后，
完整基准运行所产生的 trace、配置和配对报告才能用于讨论结果。公开论文
没有给出原始内部代码和全部端点行为，因此这些部分不能被如实重建。
