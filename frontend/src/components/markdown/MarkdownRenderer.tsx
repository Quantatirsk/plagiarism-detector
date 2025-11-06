import React from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeRaw from 'rehype-raw';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import ts from 'react-syntax-highlighter/dist/cjs/languages/prism/typescript';
import js from 'react-syntax-highlighter/dist/cjs/languages/prism/javascript';
import python from 'react-syntax-highlighter/dist/cjs/languages/prism/python';
import bash from 'react-syntax-highlighter/dist/cjs/languages/prism/bash';
import json from 'react-syntax-highlighter/dist/cjs/languages/prism/json';
import markdown from 'react-syntax-highlighter/dist/cjs/languages/prism/markdown';
import yaml from 'react-syntax-highlighter/dist/cjs/languages/prism/yaml';
import { cn } from '@/lib/utils';
import './markdown.css';

SyntaxHighlighter.registerLanguage('ts', ts);
SyntaxHighlighter.registerLanguage('typescript', ts);
SyntaxHighlighter.registerLanguage('js', js);
SyntaxHighlighter.registerLanguage('javascript', js);
SyntaxHighlighter.registerLanguage('python', python);
SyntaxHighlighter.registerLanguage('bash', bash);
SyntaxHighlighter.registerLanguage('shell', bash);
SyntaxHighlighter.registerLanguage('json', json);
SyntaxHighlighter.registerLanguage('yaml', yaml);
SyntaxHighlighter.registerLanguage('yml', yaml);
SyntaxHighlighter.registerLanguage('markdown', markdown);
SyntaxHighlighter.registerLanguage('md', markdown);

export interface MarkdownRendererProps {
  content: string;
  className?: string;
}

type CodeProps = React.HTMLAttributes<HTMLElement> & {
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
};

type TableProps = React.TableHTMLAttributes<HTMLTableElement> & {
  children?: React.ReactNode;
};

export function MarkdownRenderer({ content, className }: MarkdownRendererProps) {
  const CodeBlock = ({ inline, className, children, ...props }: CodeProps) => {
    const match = /language-(\w+)/.exec(className || '');
    const language = match?.[1];

    if (inline) {
      return (
        <code className={cn('markdown-inline-code', className)} {...props}>
          {children}
        </code>
      );
    }

    const raw = String(children ?? '').replace(/\n$/, '');

    return (
      <SyntaxHighlighter
        // @ts-expect-error - react-syntax-highlighter style type is incompatible with React.CSSProperties
        style={oneLight}
        language={language || 'text'}
        PreTag="div"
        showLineNumbers
        wrapLongLines
        {...props}
      >
        {raw}
      </SyntaxHighlighter>
    );
  };

  const TableComponent = ({ children }: TableProps) => (
    <div className="markdown-table-wrapper">
      <table>{children}</table>
    </div>
  );

  return (
    <div className={cn('markdown-renderer', className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeRaw, rehypeKatex]}
        components={{
          code: CodeBlock as Components['code'],
          table: TableComponent as Components['table'],
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

export default MarkdownRenderer;
