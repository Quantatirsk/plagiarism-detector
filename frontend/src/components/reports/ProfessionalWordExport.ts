/**
 * 专业 Word 报告生成器（修复版）
 * 使用 docx-js 生成符合商业标准的 Word 文档
 */

import {
  Document,
  Packer,
  Paragraph,
  TextRun,
  Table,
  TableRow,
  TableCell,
  HeadingLevel,
  AlignmentType,
  BorderStyle,
  WidthType,
  ShadingType,
  VerticalAlign,
  PageBreak,
  LevelFormat,
  PageOrientation
} from 'docx';

export interface WordReportData {
  title: string;
  summary: string;
  fullContent: string;
  htmlSegments?: string;
  structuredSectionsHtml?: string;
  datasetWarning?: string;
  generated_at: string;
  language: string;
}

// ========== 样式配置 ==========

const COLORS = {
  primary: '1a365d',
  secondary: '2c5282',
  accent: 'd32f2f',
  text: '2d3748',
  textLight: '718096',
  tableHeader: 'D5E8F0',
  coverBg: '1a202c',      // Dark background for cover
  coverText: 'FFFFFF'      // White text for cover
};

// ========== HTML 表格解析器 ==========

interface TableData {
  headers: string[];
  rows: string[][];
}

/**
 * 解析 HTML 表格（简化版）
 */
function parseHtmlTable(html: string): TableData | null {
  // 提取表头
  const theadMatch = html.match(/<thead>([\s\S]*?)<\/thead>/i);
  const tbodyMatch = html.match(/<tbody>([\s\S]*?)<\/tbody>/i);

  if (!theadMatch || !tbodyMatch) return null;

  // 解析表头
  const headers: string[] = [];
  const thMatches = theadMatch[1].matchAll(/<th[^>]*>([\s\S]*?)<\/th>/gi);
  for (const match of thMatches) {
    const text = match[1].replace(/<[^>]+>/g, '').trim();
    headers.push(text);
  }

  // 解析行
  const rows: string[][] = [];
  const trMatches = tbodyMatch[1].matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi);
  for (const trMatch of trMatches) {
    const row: string[] = [];
    const tdMatches = trMatch[1].matchAll(/<td[^>]*>([\s\S]*?)<\/td>/gi);
    for (const tdMatch of tdMatches) {
      const text = tdMatch[1].replace(/<[^>]+>/g, '').replace(/&nbsp;/g, ' ').trim();
      row.push(text);
    }
    if (row.length > 0) {
      rows.push(row);
    }
  }

  return { headers, rows };
}

/**
 * 创建 Word 表格
 * 直接硬编码列宽，方便调试
 */
function createTable(tableData: TableData): Table {
  // 尝试使用百分比宽度（4列：序号 15%、左侧 35%、右侧 35%、相似度 15%）
  const columnWidthsPercent = [15, 35, 35, 15]; // 百分比

  const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: 'CCCCCC' };
  const cellBorders = {
    top: tableBorder,
    bottom: tableBorder,
    left: tableBorder,
    right: tableBorder
  };

  // 表头行 - 使用百分比宽度
  const headerRow = new TableRow({
    tableHeader: true,
    cantSplit: true,
    children: tableData.headers.map((header, index) =>
      new TableCell({
        borders: cellBorders,
        width: { size: columnWidthsPercent[index], type: WidthType.PERCENTAGE },
        shading: { fill: COLORS.tableHeader, type: ShadingType.CLEAR },
        verticalAlign: VerticalAlign.CENTER,
        children: [
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { before: 40, after: 40 },
            children: [new TextRun({ text: header, font: 'SimHei', size: 20 })]
          })
        ]
      })
    )
  });

  // 数据行 - 使用百分比宽度
  const dataRows = tableData.rows.map(row =>
    new TableRow({
      children: row.map((cell, index) =>
        new TableCell({
          borders: cellBorders,
          width: { size: columnWidthsPercent[index], type: WidthType.PERCENTAGE },
          children: [
            new Paragraph({
              spacing: { before: 40, after: 40 },
              children: [new TextRun({ text: cell, size: 20 })]
            })
          ]
        })
      )
    })
  );

  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    rows: [headerRow, ...dataRows]
  });
}

// ========== Markdown 解析器 ==========

interface MarkdownBlock {
  type: 'heading1' | 'heading2' | 'heading3' | 'paragraph' | 'list-item';
  text: string;
  children?: MarkdownInline[];
}

interface MarkdownInline {
  type: 'text' | 'bold' | 'italic' | 'code';
  text: string;
}

function parseMarkdown(markdown: string): MarkdownBlock[] {
  const lines = markdown.split('\n');
  const blocks: MarkdownBlock[] = [];

  for (let line of lines) {
    line = line.trim();
    if (!line) continue;

    if (line.startsWith('### ')) {
      blocks.push({
        type: 'heading3',
        text: line.substring(4),
        children: parseInlineMarkdown(line.substring(4))
      });
    } else if (line.startsWith('## ')) {
      blocks.push({
        type: 'heading2',
        text: line.substring(3),
        children: parseInlineMarkdown(line.substring(3))
      });
    } else if (line.startsWith('# ')) {
      blocks.push({
        type: 'heading1',
        text: line.substring(2),
        children: parseInlineMarkdown(line.substring(2))
      });
    } else if (line.startsWith('- ') || line.startsWith('* ')) {
      blocks.push({
        type: 'list-item',
        text: line.substring(2),
        children: parseInlineMarkdown(line.substring(2))
      });
    } else if (/^\d+\.\s/.test(line)) {
      const match = line.match(/^\d+\.\s(.+)/);
      if (match) {
        blocks.push({
          type: 'list-item',
          text: match[1],
          children: parseInlineMarkdown(match[1])
        });
      }
    } else {
      blocks.push({
        type: 'paragraph',
        text: line,
        children: parseInlineMarkdown(line)
      });
    }
  }

  return blocks;
}

function parseInlineMarkdown(text: string): MarkdownInline[] {
  const parts: MarkdownInline[] = [];
  let current = '';
  let i = 0;

  while (i < text.length) {
    if (text[i] === '*' && text[i + 1] === '*') {
      if (current) {
        parts.push({ type: 'text', text: current });
        current = '';
      }
      i += 2;
      let boldText = '';
      while (i < text.length && !(text[i] === '*' && text[i + 1] === '*')) {
        boldText += text[i];
        i++;
      }
      if (boldText) {
        parts.push({ type: 'bold', text: boldText });
      }
      i += 2;
    } else if (text[i] === '*') {
      if (current) {
        parts.push({ type: 'text', text: current });
        current = '';
      }
      i++;
      let italicText = '';
      while (i < text.length && text[i] !== '*') {
        italicText += text[i];
        i++;
      }
      if (italicText) {
        parts.push({ type: 'italic', text: italicText });
      }
      i++;
    } else if (text[i] === '`') {
      if (current) {
        parts.push({ type: 'text', text: current });
        current = '';
      }
      i++;
      let codeText = '';
      while (i < text.length && text[i] !== '`') {
        codeText += text[i];
        i++;
      }
      if (codeText) {
        parts.push({ type: 'code', text: codeText });
      }
      i++;
    } else {
      current += text[i];
      i++;
    }
  }

  if (current) {
    parts.push({ type: 'text', text: current });
  }

  return parts.length > 0 ? parts : [{ type: 'text', text: text }];
}

function markdownBlocksToParagraphs(blocks: MarkdownBlock[], listRef: string = 'default-list'): Paragraph[] {
  const paragraphs: Paragraph[] = [];

  for (const block of blocks) {
    const children = (block.children || []).map(inline => {
      switch (inline.type) {
        case 'bold':
          return new TextRun({ text: inline.text, bold: true });
        case 'italic':
          return new TextRun({ text: inline.text, italics: true });
        case 'code':
          return new TextRun({ text: inline.text, font: 'Courier New', size: 20 });
        default:
          return new TextRun({ text: inline.text });
      }
    });

    switch (block.type) {
      case 'heading1':
        paragraphs.push(
          new Paragraph({
            heading: HeadingLevel.HEADING_1,
            children: children.length > 0 ? children : [new TextRun(block.text)]
          })
        );
        break;
      case 'heading2':
        paragraphs.push(
          new Paragraph({
            heading: HeadingLevel.HEADING_2,
            children: children.length > 0 ? children : [new TextRun(block.text)]
          })
        );
        break;
      case 'heading3':
        paragraphs.push(
          new Paragraph({
            heading: HeadingLevel.HEADING_3,
            children: children.length > 0 ? children : [new TextRun(block.text)]
          })
        );
        break;
      case 'list-item':
        paragraphs.push(
          new Paragraph({
            numbering: { reference: listRef, level: 0 },
            children: children.length > 0 ? children : [new TextRun(block.text)]
          })
        );
        break;
      default:
        paragraphs.push(
          new Paragraph({
            children: children.length > 0 ? children : [new TextRun(block.text)],
            spacing: { before: 100, after: 100 }
          })
        );
    }
  }

  return paragraphs;
}

// ========== 封面页生成 ==========

function createCoverPage(data: WordReportData): Table[] {
  const reportDate = new Date(data.generated_at).toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });

  // 创建深色背景封面表格
  const coverTable = new Table({
    width: { size: 85, type: WidthType.PERCENTAGE },
    alignment: AlignmentType.CENTER,
    borders: {
      top: { style: BorderStyle.NONE, size: 0 },
      bottom: { style: BorderStyle.NONE, size: 0 },
      left: { style: BorderStyle.NONE, size: 0 },
      right: { style: BorderStyle.NONE, size: 0 },
      insideHorizontal: { style: BorderStyle.NONE, size: 0 },
      insideVertical: { style: BorderStyle.NONE, size: 0 }
    },
    rows: [
      new TableRow({
        height: { value: 14400, rule: 'atLeast' }, // ~10 inches tall
        children: [
          new TableCell({
            shading: { fill: COLORS.coverBg, type: ShadingType.CLEAR },
            verticalAlign: VerticalAlign.CENTER,
            width: { size: 100, type: WidthType.PERCENTAGE },
            margins: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
            children: [
              // 顶部标识
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 2400, after: 1200 },
                children: [
                  new TextRun({
                    text: '招投标雷同性分析报告',
                    size: 28,
                    color: COLORS.coverText
                  })
                ]
              }),

              // 主标题
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 1200, after: 800 },
                children: [
                  new TextRun({
                    text: data.title,
                    size: 44,
                    bold: true,
                    color: COLORS.coverText
                  })
                ]
              }),

              // 副标题
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { after: 2400 },
                children: [
                  new TextRun({
                    text: 'Procurement Compliance Intelligence Report',
                    size: 24,
                    color: COLORS.coverText,
                    italics: true
                  })
                ]
              }),

              // 空白占位，用于垂直居中
              new Paragraph({
                spacing: { before: 3600 },
                children: [new TextRun({ text: '', size: 1 })]
              }),

              // 元信息
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 400 },
                children: [
                  new TextRun({
                    text: `报告日期：${reportDate}`,
                    size: 22,
                    color: COLORS.coverText
                  })
                ]
              }),
              new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { before: 200, after: 400 },
                children: [
                  new TextRun({
                    text: `语言：${data.language === 'zh' ? '中文' : 'English'}`,
                    size: 22,
                    color: COLORS.coverText
                  })
                ]
              })
            ]
          })
        ]
      })
    ]
  });

  return [coverTable];
}

// ========== 目录页生成 ==========

function createTableOfContents(hasSegmentsTable: boolean): Paragraph[] {
  return [
    // 目录标题 - 更大更醒目
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 600, after: 400 },
      children: [
        new TextRun({
          text: '目  录',
          font: 'SimHei',
          size: 48,
          bold: true,
          color: COLORS.primary
        })
      ]
    }),

    // 装饰性分隔线
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 100, after: 600 },
      border: {
        bottom: {
          color: COLORS.primary,
          space: 1,
          style: BorderStyle.SINGLE,
          size: 12
        }
      },
      children: [new TextRun({ text: '', size: 2 })]
    }),

    // 目录项 1
    new Paragraph({
      spacing: { before: 300, after: 200 },
      indent: { left: 720 },
      children: [
        new TextRun({
          text: '1.',
          font: 'SimHei',
          size: 32,
          bold: true,
          color: COLORS.primary
        }),
        new TextRun({
          text: ' 报告摘要',
          font: 'SimHei',
          size: 32,
          color: COLORS.text
        })
      ]
    }),

    // 目录项 2
    new Paragraph({
      spacing: { before: 200, after: 200 },
      indent: { left: 720 },
      children: [
        new TextRun({
          text: '2.',
          font: 'SimHei',
          size: 32,
          bold: true,
          color: COLORS.primary
        }),
        new TextRun({
          text: ' 详细分析',
          font: 'SimHei',
          size: 32,
          color: COLORS.text
        })
      ]
    }),

    // 目录项 3（条件显示）
    ...(hasSegmentsTable
      ? [
          new Paragraph({
            spacing: { before: 200, after: 200 },
            indent: { left: 720 },
            children: [
              new TextRun({
                text: '3.',
                font: 'SimHei',
                size: 32,
                bold: true,
                color: COLORS.primary
              }),
              new TextRun({
                text: ' 相似片段清单',
                font: 'SimHei',
                size: 32,
                color: COLORS.text
              })
            ]
          })
        ]
      : []),

    // 底部留白
    new Paragraph({
      spacing: { before: 600 },
      children: [new TextRun({ text: '', size: 2 })]
    })
  ];
}

// ========== 主导出函数 ==========

export async function generateProfessionalWordDocument(data: WordReportData): Promise<Blob> {
  const summaryBlocks = parseMarkdown(data.summary);
  const contentBlocks = parseMarkdown(data.fullContent);

  // 解析 HTML 表格
  let segmentsTable: Table | null = null;
  if (data.htmlSegments) {
    const tableData = parseHtmlTable(data.htmlSegments);
    if (tableData) {
      segmentsTable = createTable(tableData);
    }
  }

  const doc = new Document({
    styles: {
      default: {
        document: {
          run: {
            font: 'Arial',
            size: 24
          },
          paragraph: {
            spacing: { line: 276, before: 100, after: 100 } // 单倍行距
          }
        }
      },
      paragraphStyles: [
        {
          id: 'Heading1',
          name: 'Heading 1',
          basedOn: 'Normal',
          next: 'Normal',
          quickFormat: true,
          run: {
            font: 'SimHei',
            size: 32,
            color: COLORS.primary
          },
          paragraph: {
            spacing: { line: 276, before: 400, after: 200 },
            outlineLevel: 0
          }
        },
        {
          id: 'Heading2',
          name: 'Heading 2',
          basedOn: 'Normal',
          next: 'Normal',
          quickFormat: true,
          run: {
            font: 'SimHei',
            size: 28,
            color: COLORS.secondary
          },
          paragraph: {
            spacing: { line: 276, before: 300, after: 150 },
            outlineLevel: 1
          }
        },
        {
          id: 'Heading3',
          name: 'Heading 3',
          basedOn: 'Normal',
          next: 'Normal',
          quickFormat: true,
          run: {
            font: 'SimHei',
            size: 24,
            color: COLORS.secondary
          },
          paragraph: {
            spacing: { line: 276, before: 200, after: 100 },
            outlineLevel: 2
          }
        }
      ]
    },
    numbering: {
      config: [
        {
          reference: 'default-list',
          levels: [
            {
              level: 0,
              format: LevelFormat.BULLET,
              text: '•',
              alignment: AlignmentType.LEFT,
              style: {
                paragraph: {
                  indent: { left: 360, hanging: 360 }
                }
              }
            }
          ]
        }
      ]
    },
    sections: [
      // 封面页（无边距，深色背景）
      {
        properties: {
          page: {
            margin: { top: 0, right: 0, bottom: 0, left: 0 }
          }
        },
        children: [...createCoverPage(data)]
      },

      // 主内容区（纵向）
      {
        properties: {
          page: {
            margin: { top: 900, right: 900, bottom: 900, left: 900 }
          }
        },
        children: [
          ...createTableOfContents(!!segmentsTable),
          new Paragraph({ children: [new PageBreak()] }),

          new Paragraph({
            heading: HeadingLevel.HEADING_1,
            children: [new TextRun('报告摘要')]
          }),
          ...markdownBlocksToParagraphs(summaryBlocks),

          ...(data.datasetWarning
            ? [
                new Paragraph({
                  spacing: { before: 400, after: 200 },
                  children: [
                    new TextRun({
                      text: '⚠️ 重要提示',
                      bold: true,
                      color: COLORS.accent,
                      size: 22
                    })
                  ]
                }),
                new Paragraph({
                  children: [new TextRun(data.datasetWarning)],
                  spacing: { before: 100, after: 400 }
                })
              ]
            : []),

          new Paragraph({ children: [new PageBreak()] }),

          new Paragraph({
            heading: HeadingLevel.HEADING_1,
            children: [new TextRun('详细分析')]
          }),
          ...markdownBlocksToParagraphs(contentBlocks)
        ]
      },

      // 高相似片段清单区（横向，便于表格展示）
      ...(segmentsTable
        ? [
            {
              properties: {
                page: {
                  margin: { top: 900, right: 900, bottom: 900, left: 900 },
                  size: {
                    orientation: PageOrientation.LANDSCAPE
                  }
                }
              },
              children: [
                new Paragraph({
                  heading: HeadingLevel.HEADING_1,
                  children: [new TextRun('高相似片段清单')]
                }),
                new Paragraph({
                  spacing: { before: 200, after: 200 },
                  children: [new TextRun('')]
                }),
                segmentsTable
              ]
            }
          ]
        : [])
    ]
  });

  const blob = await Packer.toBlob(doc);
  return blob;
}
