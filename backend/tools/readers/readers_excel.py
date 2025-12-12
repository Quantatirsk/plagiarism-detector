"""
Excel 解析器 - 支持 xlsx, xls, xlsb, ods 格式
使用 python-calamine 生成 HTML table 格式输出，优化 LLM 解析
"""

from typing import Optional, Dict, Tuple
from datetime import datetime, date, timedelta
from python_calamine import CalamineWorkbook
from html import escape
from .readers_base import BaseParser
import logging


# 配置日志记录器
logger = logging.getLogger(__name__)
class ExcelParser(BaseParser):
    """
    统一的 Excel 解析器，支持多种表格格式。
    
    支持格式: xlsx, xls, xlsb, ods
    输出格式: 最小化 HTML table，优化 LLM 解析
    """
    
    def parse(self, file_path: str) -> Optional[str]:
        """
        解析 Excel 文件并生成 HTML table。
        
        Args:
            file_path: Excel 文件路径
            
        Returns:
            HTML table 格式的内容，解析失败返回 None
        """
        
        try:
            workbook = CalamineWorkbook.from_path(file_path)
            html = []

            for sheet_name in workbook.sheet_names:
                sheet = workbook.get_sheet_by_name(sheet_name)
                data = sheet.to_python()
                
                if not data:
                    continue
                
                # Sheet 名称作为 H2 标题
                html.append(f'<h2>{escape(sheet_name)}</h2>')
                html.append('<table>')

                # 处理合并单元格
                merged_map: Dict[Tuple[int, int], Dict[str, int]] = {}
                if hasattr(sheet, 'merged_cell_ranges'):
                    merged_ranges = sheet.merged_cell_ranges
                    if merged_ranges:  # 确保不是 None
                        for merge in merged_ranges:
                            if len(merge) == 2:
                                start, end = merge
                                merged_map[(start[0], start[1])] = {
                                    'rowspan': end[0] - start[0] + 1,
                                    'colspan': end[1] - start[1] + 1,
                                    'end_row': end[0],
                                    'end_col': end[1]
                                }
                
                skip_cells = set()
                max_cols = max(len(r) for r in data) if data else 0
                
                for row_idx, row in enumerate(data):
                    html.append('<tr>')
                    
                    for col_idx in range(max_cols):
                        if (row_idx, col_idx) in skip_cells:
                            continue
                        
                        value = row[col_idx] if col_idx < len(row) else None
                        merge = merged_map.get((row_idx, col_idx))
                        
                        # 构建最小属性
                        attrs = []
                        if merge:
                            if merge['rowspan'] > 1:
                                attrs.append(f'rowspan="{merge["rowspan"]}"')
                            if merge['colspan'] > 1:
                                attrs.append(f'colspan="{merge["colspan"]}"')
                            
                            # 标记需要跳过的单元格
                            for r in range(row_idx, merge['end_row'] + 1):
                                for c in range(col_idx, merge['end_col'] + 1):
                                    if r != row_idx or c != col_idx:
                                        skip_cells.add((r, c))
                        
                        # 格式化值
                        text = self._format_value(value)
                        
                        # 简单标签选择 - 第一行作为表头
                        tag = 'th' if row_idx == 0 else 'td'
                        attr_str = ' ' + ' '.join(attrs) if attrs else ''
                        html.append(f'<{tag}{attr_str}>{text}</{tag}>')
                    
                    html.append('</tr>')
                
                html.append('</table>')
            
            return '\n'.join(html) if html else None
            
        except Exception as e:
            logger.error(f"解析 Excel 文件错误 {file_path}: {e}")
            return None
    
    def _format_value(self, value) -> str:
        """
        格式化单元格值为字符串。
        
        Args:
            value: 单元格原始值
            
        Returns:
            格式化后的字符串
        """
        if value is None:
            return ''
        
        # Excel 日期序列号转换为可读日期
        if isinstance(value, float) and 25000 <= value <= 50000 and value == int(value):
            try:
                # Excel 日期从 1900-01-01 开始，但有 1900 年闰年 bug
                date_val = datetime(1900, 1, 1) + timedelta(days=int(value) - 2)
                return date_val.strftime('%Y-%m-%d')
            except (ValueError, OverflowError):
                pass
        
        # 清晰的数字显示
        if isinstance(value, float):
            return str(int(value)) if value == int(value) else f'{value:.2f}'
        
        # 日期时间格式化
        if isinstance(value, datetime):
            return value.strftime('%Y-%m-%d %H:%M')
        if isinstance(value, date):
            return value.strftime('%Y-%m-%d')
        
        # 转义 HTML 特殊字符
        return escape(str(value))