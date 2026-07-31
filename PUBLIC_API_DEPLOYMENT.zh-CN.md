# 公网 Add/Search API 部署

本文件用于将 Mi-Memory 部署为平台可提交的 HTTPS 接口。部署使用论文
运行时：每次 Add/Search 会调用已配置的模型端点；密钥只保存在服务器的
`.env`，绝不提交到 Git。

## 前置条件

1. 一台可长期运行 Docker Compose 的公网 Linux 主机；开放 TCP `80` 与 `443`。
2. 一个域名，例如 `memory.example.com`，其 DNS A/AAAA 记录指向该主机。
3. 服务器上的 `.env` 已配置模型端点，且该文件权限仅限部署账户读取。

## 启动

在仓库根目录执行。将示例域名与 Token 替换为自己的值；Token 必须随机、
至少 32 字符，且只填入平台一次。

```bash
export PUBLIC_DOMAIN=memory.example.com
export MIMEMORY_API_TOKEN='replace-with-a-long-random-token'
docker compose -f compose.public.yaml up -d --build
```

Caddy 在 DNS 和端口均可达后自动申请 TLS 证书。验证：

```bash
curl https://memory.example.com/health
```

预期为 `{"status":"ok","service":"mi-memory-add-search"}`。服务数据保存在
Docker 卷 `mimemory-public-data`，不会写入镜像或仓库。

## 平台表单

| 字段 | 填写内容 |
| --- | --- |
| Add API 地址 | `https://memory.example.com/v1/add` |
| Search API 地址 | `https://memory.example.com/v1/search` |
| 认证方式 | `Authorization: Token` |
| 记忆系统 Key | 与 `MIMEMORY_API_TOKEN` 相同的值 |

也兼容 `Authorization: Bearer <token>` 和 `X-API-Key: <token>`。平台提交前应
确认 `/health`、`/v1/add`、`/v1/search` 均通过 HTTPS 可从公网访问。不要提交
本机 `192.168.*`、`localhost` 地址或模型提供商密钥。

## 运行边界

- Add 同步完成抽取、嵌入和持久化后才返回成功；Search 使用规划、混合召回和
  严格重排。
- 按平台要求保留评测服务 30 天，并在窗口结束后运行项目的 retention 清理命令。
- `.env` 与 Docker 数据卷含有运行凭据或记忆数据，均不应共享或提交。
