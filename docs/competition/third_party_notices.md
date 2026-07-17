# 第三方软件、外部服务与 AI 工具说明

## 1. 适用范围与核验方法

项目自身代码采用根目录 [MIT License](../../LICENSE)。该许可证不改变第三方软件、模型服务、网页内容、课程资料、字体/媒体和容器系统包各自的许可与条款。

下表按 [pyproject.toml](../../pyproject.toml) 和
[frontend/package.json](../../frontend/package.json) 的直接依赖整理，许可证是 2026-07-17 根据上游项目/包元数据形成的审阅快照。确切发行物应以实际 lock、wheel/npm 包内许可证和容器 SBOM 为准；表格不是法律意见，也不替代许可证负责人审批。

## 2. Python 运行时直接依赖

| 名称 | 来源 | 上游声明许可 |
| --- | --- | --- |
| `langchain`、`langchain-openai`、`langchain-community`、`langchain-text-splitters` | [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | MIT |
| `langgraph`、`langgraph-checkpoint-postgres` | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | MIT |
| `psycopg[binary,pool]` | [psycopg/psycopg](https://github.com/psycopg/psycopg) | LGPL-3.0-only；binary/libpq 等实际发行组件另行核验 |
| `langchain-chroma` | [LangChain integrations](https://github.com/langchain-ai/langchain) | MIT |
| `chromadb` | [chroma-core/chroma](https://github.com/chroma-core/chroma) | Apache-2.0 |
| `pymupdf` | [pymupdf/PyMuPDF](https://github.com/pymupdf/PyMuPDF) | AGPL-3.0 或另购商业许可 |
| `rank-bm25` | [dorianbrown/rank_bm25](https://github.com/dorianbrown/rank_bm25) | Apache-2.0 |
| `jieba` | [fxsjy/jieba](https://github.com/fxsjy/jieba) | MIT |
| `xmind` | [zhuifengshen/xmind](https://github.com/zhuifengshen/xmind) | MIT |
| `python-docx` | [python-openxml/python-docx](https://github.com/python-openxml/python-docx) | MIT |
| `playwright` | [microsoft/playwright-python](https://github.com/microsoft/playwright-python) | Apache-2.0 |
| `pillow` | [python-pillow/Pillow](https://github.com/python-pillow/Pillow) | HPND（Historical Permission Notice and Disclaimer） |
| `imageio` | [imageio/imageio](https://github.com/imageio/imageio) | BSD-2-Clause |
| `python-dotenv` | [theskumar/python-dotenv](https://github.com/theskumar/python-dotenv) | BSD-3-Clause |
| `pyyaml` | [yaml/pyyaml](https://github.com/yaml/pyyaml) | MIT |
| `httpx` | [encode/httpx](https://github.com/encode/httpx) | BSD-3-Clause |
| `pydantic` | [pydantic/pydantic](https://github.com/pydantic/pydantic) | MIT |
| `typing-extensions` | [python/typing_extensions](https://github.com/python/typing_extensions) | PSF-2.0 |
| `fastapi` | [fastapi/fastapi](https://github.com/fastapi/fastapi) | MIT |
| `uvicorn` | [encode/uvicorn](https://github.com/encode/uvicorn) | BSD-3-Clause |
| `aiosqlite` | [omnilib/aiosqlite](https://github.com/omnilib/aiosqlite) | MIT |
| `opentelemetry-api`、`opentelemetry-sdk`、OTLP gRPC exporter、FastAPI instrumentation | [open-telemetry/opentelemetry-python](https://github.com/open-telemetry/opentelemetry-python) 与 contrib | Apache-2.0 |

## 3. 前端运行时直接依赖

| 名称 | 来源 | 上游声明许可 |
| --- | --- | --- |
| `@dagrejs/dagre` | [dagrejs/dagre](https://github.com/dagrejs/dagre) | MIT |
| `@radix-ui/react-popover`、`react-scroll-area`、`react-slot`、`react-tooltip` | [radix-ui/primitives](https://github.com/radix-ui/primitives) | MIT |
| `@xyflow/react` | [xyflow/xyflow](https://github.com/xyflow/xyflow) | MIT |
| `class-variance-authority` | [joe-bell/cva](https://github.com/joe-bell/cva) | Apache-2.0 |
| `clsx` | [lukeed/clsx](https://github.com/lukeed/clsx) | MIT |
| `lucide-react` | [lucide-icons/lucide](https://github.com/lucide-icons/lucide) | ISC |
| `mermaid` | [mermaid-js/mermaid](https://github.com/mermaid-js/mermaid) | MIT |
| `next` | [vercel/next.js](https://github.com/vercel/next.js) | MIT |
| `next-themes` | [pacocoursey/next-themes](https://github.com/pacocoursey/next-themes) | MIT |
| `react`、`react-dom` | [facebook/react](https://github.com/facebook/react) | MIT |
| `react-markdown` | [remarkjs/react-markdown](https://github.com/remarkjs/react-markdown) | MIT |
| `recharts` | [recharts/recharts](https://github.com/recharts/recharts) | MIT |
| `remark-gfm` | [remarkjs/remark-gfm](https://github.com/remarkjs/remark-gfm) | MIT |
| `tailwind-merge` | [dcastil/tailwind-merge](https://github.com/dcastil/tailwind-merge) | MIT |

## 4. 直接开发与质量工具

| 名称 | 来源 | 上游声明许可 |
| --- | --- | --- |
| `pytest` | [pytest-dev/pytest](https://github.com/pytest-dev/pytest) | MIT |
| `pytest-asyncio` | [pytest-dev/pytest-asyncio](https://github.com/pytest-dev/pytest-asyncio) | Apache-2.0 |
| `bandit` | [PyCQA/bandit](https://github.com/PyCQA/bandit) | Apache-2.0 |
| `import-linter` | [seddonym/import-linter](https://github.com/seddonym/import-linter) | BSD-2-Clause |
| `mypy` | [python/mypy](https://github.com/python/mypy) | MIT |
| `ruff` | [astral-sh/ruff](https://github.com/astral-sh/ruff) | MIT |
| `vulture` | [jendrikseipp/vulture](https://github.com/jendrikseipp/vulture) | MIT |
| `@tailwindcss/postcss`、`tailwindcss` | [tailwindlabs/tailwindcss](https://github.com/tailwindlabs/tailwindcss) | MIT |
| `@testing-library/jest-dom`、`@testing-library/react` | [Testing Library](https://github.com/testing-library) | MIT |
| `@types/node`、`@types/react`、`@types/react-dom` | [DefinitelyTyped](https://github.com/DefinitelyTyped/DefinitelyTyped) | MIT（各 definition 包须按其包内文件复核） |
| `@typescript-eslint/parser` | [typescript-eslint/typescript-eslint](https://github.com/typescript-eslint/typescript-eslint) | MIT |
| `autoprefixer` | [postcss/autoprefixer](https://github.com/postcss/autoprefixer) | MIT |
| `eslint` | [eslint/eslint](https://github.com/eslint/eslint) | MIT |
| `jsdom` | [jsdom/jsdom](https://github.com/jsdom/jsdom) | MIT |
| `postcss` | [postcss/postcss](https://github.com/postcss/postcss) | MIT |
| `typescript` | [microsoft/TypeScript](https://github.com/microsoft/TypeScript) | Apache-2.0 |
| `vitest` | [vitest-dev/vitest](https://github.com/vitest-dev/vitest) | MIT |

## 5. 发行前必须人工确认的高风险项

### PyMuPDF

`pymupdf>=1.27,<2` 是直接运行时依赖。上游提供 AGPL-3.0 与商业许可两条路径。项目根 MIT 许可证不会消除 AGPL 或商业许可义务。对外分发源代码、镜像、桌面包或提供网络服务前，许可证负责人必须根据实际使用与分发模式选择并留存合规依据；未确认前不得宣称整个发行物“仅 MIT”。

### Psycopg、浏览器和媒体组件

Psycopg 声明 LGPL-3.0-only，binary extra 及其捆绑/动态链接组件需要按实际 wheel 复核。Docker 镜像中的 Chromium、ffmpeg、系统库、字体和编解码器不在 pip/npm 直接依赖表内，必须从最终镜像生成 SBOM、保留 notice，并检查 ffmpeg 的具体构建选项和媒体专利/许可范围。

### 课程资料、索引和生成内容

课程 PDF/文档、网页材料、图片、字体、视频素材以及由其派生的 Parent–Child 索引可能受版权、数据库权或使用条款约束。技术上能读取不等于有权提交、公开演示或再分发。干净 Git checkout 故意不承诺自包含这些资产；资料负责人必须记录来源、权利基础、允许用途、保留期限和删除流程。

## 6. 外部服务与模型

运行时可通过严格配置使用对话模型、embedding、reranker 和网页研究服务。相关 API key 只能存在于忽略的 `.env` 或 secret store；文档、日志和提交不得包含值、Authorization、完整数据库 URI 或 Provider body。

外部服务通常不是随仓库再分发的开源组件，其服务条款、隐私政策、数据驻留、训练使用、配额和内容政策必须由部署者单独接受。环境变量名或兼容协议不等于对某家服务的许可授权。

## 7. AI 辅助开发披露

- 可审计仓库记录显示开发过程中使用了 [OpenAI Codex](https://openai.com/codex/) 进行代码审阅、实现和文档辅助。Codex 是外部 AI 服务，不受本项目 MIT 许可证覆盖；开发者对采纳的 diff、测试和最终提交负责。
- 当前仓库没有足够证据证明开发过程中使用过科大讯飞 AI Coding 工具。赛题“实现条件”对其他 AI 辅助工具提出了科大讯飞相关要求；参赛负责人必须向组委会确认适用口径并提供真实证据。不得在本说明中虚构工具使用。
- 运行时模型/检索服务与开发期 AI 工具是两类不同事项，不能互相代替合规证明。

## 8. 维护要求

依赖升级、容器基础镜像变化、增加素材或更换外部服务时必须同步更新本文件，并以实际 lock 和镜像生成新的依赖清单/SBOM。提交包应包含所需许可证文本和 notice；无法确认许可的组件或资产应在发布前移除或取得授权，而不是用免责声明替代。
