"use client"

import { useEffect, useState } from "react"
import ReactMarkdown, { type Components } from "react-markdown"
import remarkGfm from "remark-gfm"

const PRINT_PAYLOAD_KEY = "review_doc_print_payload"

interface PrintPayload {
  title: string
  markdown: string
}

const fallbackPayload: PrintPayload = {
  title: "Markdown 复习文档",
  markdown: "# Markdown 复习文档\n\n未找到可打印的复习文档内容。",
}

const markdownComponents: Components = {
  h1: ({ children }) => <h1>{children}</h1>,
  h2: ({ children }) => <h2>{children}</h2>,
  h3: ({ children }) => <h3>{children}</h3>,
  p: ({ children }) => <p>{children}</p>,
  ul: ({ children }) => <ul>{children}</ul>,
  ol: ({ children }) => <ol>{children}</ol>,
  li: ({ children }) => <li>{children}</li>,
  code: ({ className, children }) => {
    const isBlock = className?.startsWith("language-")
    return isBlock ? <code className="code-block">{children}</code> : <code>{children}</code>
  },
  pre: ({ children }) => <pre>{children}</pre>,
  table: ({ children }) => <table>{children}</table>,
  th: ({ children }) => <th>{children}</th>,
  td: ({ children }) => <td>{children}</td>,
}

export default function ReviewDocPrintPage() {
  const [payload, setPayload] = useState<PrintPayload>(fallbackPayload)

  useEffect(() => {
    const raw = window.sessionStorage.getItem(PRINT_PAYLOAD_KEY)
    if (!raw) return
    try {
      const parsed = JSON.parse(raw)
      setPayload({
        title: typeof parsed.title === "string" && parsed.title.trim() ? parsed.title : fallbackPayload.title,
        markdown:
          typeof parsed.markdown === "string" && parsed.markdown.trim()
            ? parsed.markdown
            : fallbackPayload.markdown,
      })
    } catch {
      setPayload(fallbackPayload)
    }
  }, [])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      window.print()
    }, 500)
    return () => window.clearTimeout(timer)
  }, [payload.markdown])

  return (
    <main className="print-page">
      <div className="print-toolbar">
        <button type="button" onClick={() => window.print()}>
          打印 / 保存为 PDF
        </button>
        <button type="button" onClick={() => window.close()}>
          关闭
        </button>
      </div>

      <article className="print-document" aria-label={payload.title}>
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
          {payload.markdown}
        </ReactMarkdown>
      </article>

      <style>{`
        html,
        body {
          margin: 0;
          background: #ffffff;
          color: #111111;
        }

        .print-page {
          min-height: 100vh;
          background: #ffffff;
          color: #111111;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
        }

        .print-toolbar {
          position: sticky;
          top: 0;
          z-index: 10;
          display: flex;
          justify-content: flex-end;
          gap: 8px;
          padding: 12px 24px;
          border-bottom: 1px solid #d9d9d9;
          background: #ffffff;
        }

        .print-toolbar button {
          border: 1px solid #222222;
          border-radius: 6px;
          background: #ffffff;
          color: #111111;
          padding: 7px 12px;
          font-size: 13px;
          cursor: pointer;
        }

        .print-toolbar button:hover {
          background: #f2f2f2;
        }

        .print-document {
          width: min(100%, 794px);
          margin: 0 auto;
          padding: 28px 36px 56px;
          box-sizing: border-box;
          line-height: 1.68;
          font-size: 14px;
        }

        .print-document h1 {
          margin: 0 0 20px;
          font-size: 28px;
          line-height: 1.25;
          color: #000000;
        }

        .print-document h2 {
          margin: 24px 0 10px;
          padding-bottom: 5px;
          border-bottom: 1px solid #cccccc;
          font-size: 20px;
          line-height: 1.35;
          color: #000000;
        }

        .print-document h3 {
          margin: 18px 0 8px;
          font-size: 16px;
          line-height: 1.4;
          color: #000000;
        }

        .print-document p {
          margin: 0 0 10px;
        }

        .print-document ul,
        .print-document ol {
          margin: 0 0 12px 24px;
          padding: 0;
        }

        .print-document li {
          margin: 4px 0;
        }

        .print-document table {
          width: 100%;
          margin: 14px 0 18px;
          border-collapse: collapse;
          font-size: 13px;
          page-break-inside: avoid;
        }

        .print-document th,
        .print-document td {
          border: 1px solid #777777;
          padding: 6px 8px;
          vertical-align: top;
          text-align: left;
        }

        .print-document th {
          background: #eeeeee;
          font-weight: 700;
        }

        .print-document code {
          border-radius: 3px;
          background: #eeeeee;
          padding: 1px 4px;
          font-family: Consolas, "SFMono-Regular", monospace;
          font-size: 0.92em;
        }

        .print-document pre {
          overflow-x: auto;
          border-radius: 6px;
          background: #eeeeee;
          padding: 10px;
          white-space: pre-wrap;
          page-break-inside: avoid;
        }

        .print-document pre code,
        .print-document code.code-block {
          display: block;
          background: transparent;
          padding: 0;
        }

        @page {
          size: A4;
          margin: 16mm;
        }

        @media print {
          .print-toolbar {
            display: none;
          }

          .print-document {
            width: auto;
            margin: 0;
            padding: 0;
          }
        }
      `}</style>
    </main>
  )
}
