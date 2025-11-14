/**
 * 报告导出工具函数
 * 独立文件以支持 React Fast Refresh
 */
import { saveAs } from 'file-saver';
import { generateProfessionalWordDocument, type WordReportData } from './ProfessionalWordExport';
import type { ReportData } from './ReportViewer';

/**
 * 导出报告为 Word 文档
 */
export async function exportReportToWord(report: ReportData): Promise<void> {
  const { title, content, generated_at, language } = report;
  const structured = content?.structured_report;
  const fullContent = content?.full_content || '';
  const datasetWarning = structured?.summary?.large_dataset_warning || structured?.large_dataset_warning;
  const safeTitle = title.replace(/[^\w\u4e00-\u9fa5]+/g, '-');

  // 直接使用原始 HTML，Word 导出模块会自行解析
  const wordData: WordReportData = {
    title,
    summary: report.summary || '暂无摘要',
    fullContent: fullContent || '暂无详细内容',
    htmlSegments: structured?.summary?.segments_table_html,
    structuredSectionsHtml: undefined, // Word 导出暂不使用此字段
    datasetWarning,
    generated_at,
    language
  };

  try {
    const blob = await generateProfessionalWordDocument(wordData);
    saveAs(blob, `${safeTitle || 'report'}.docx`);
  } catch (error) {
    console.error('导出 Word 报告失败', error);
    throw error;
  }
}
