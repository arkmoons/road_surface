# MLOps Final Project

> 占位说明：应用主题与机器学习模型待定（等数据集确定后更新本文件）。

## 目录结构

```
app/                  # 应用服务（API 入口 main.py 等）
ml/                   # 训练脚本、数据处理、实验配置
  configs/            # 超参数 / 训练配置
  registry/           # 当前生产模型说明 (current_model.md)
data/                 # 训练数据（由 DVC 追踪，原始数据不入 Git）
tests/                # 自动化测试
.github/workflows/    # CI/CD 流水线
```

## 分支策略

- `main`     —— 生产分支（合并触发生产镜像构建）
- `staging`  —— 预发布分支（合并触发 staging 镜像构建）
- `dev`      —— 开发集成分支
- `feature/*`—— 功能分支（从 dev 切出，合回 dev）

合并路径：`feature/* → dev → staging → main`

## 技术栈（计划）

- 应用：Python + FastAPI
- 机器学习：待定（数据集确定后补充）
- 数据版本控制：DVC + Azure Blob Storage
- 实验追踪：MLflow
- 容器：Docker
- CI/CD：GitHub Actions
- 测试：pytest（单元 / 集成 / API / E2E）
