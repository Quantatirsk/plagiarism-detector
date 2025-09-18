"""
使用直接XML解析的DOCX解析器。

遵循技术报告中关于使用lxml + zipfile的建议，
通过绕过高级API开销来实现最快的DOCX文本提取。
"""

from typing import Optional
from lxml import etree
import zipfile
from .readers_base import BaseParser
import logging


# 配置日志记录器
logger = logging.getLogger(__name__)


class DOCXParser(BaseParser):
    """
    使用直接XML解析的高性能DOCX解析器。

    正如技术报告中所建议，这种方法通过使用lxml的C实现
    直接解析底层XML结构，绕过了python-docx高级API的开销。
    """

    def parse(self, file_path: str) -> Optional[str]:
        """
        通过直接从 XML 中提取文本来解析 DOCX 文件。

        Args:
            file_path: DOCX文件路径

        Returns:
            提取的纯文本，解析失败时返回None
        """
        try:
            # 将DOCX文件作为ZIP存档打开
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                # 读取主文档XML
                try:
                    xml_content = zip_file.read('word/document.xml')
                except KeyError:
                    logger.info(f"无效的DOCX文件：缺少word/document.xml，文件为 {file_path}")
                    return None

                # 使用lxml解析XML（基于C，非常快）
                root = etree.fromstring(xml_content)

                # 为Word文档定义命名空间
                namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

                # 提取段落而不是单个文本节点
                paragraphs = root.xpath('//w:p', namespaces=namespace)

                # 从每个段落中提取文本
                paragraph_texts = []
                for paragraph in paragraphs:
                    # 获取此段落内的所有文本节点
                    text_nodes = paragraph.xpath('.//w:t/text()', namespaces=namespace)
                    paragraph_text = ''.join(text_nodes).strip()

                    if paragraph_text:  # 只添加非空段落
                        paragraph_texts.append(paragraph_text)

                # 用双换行符连接段落以清晰分隔
                return '\n\n'.join(paragraph_texts)

        except Exception as e:
            logger.error(f"解析DOCX文件错误 {file_path}: {e}")
            return None

    # 不再需要 get_supported_extensions() - 由 readers_parser_map.py 管理
    # Linux 哲学：单一职责，只负解析