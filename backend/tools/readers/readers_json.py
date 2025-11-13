"""
JSON 解析器 - 支持修复功能
"""

import json
import logging
from typing import Optional
from pathlib import Path
from json_repair import repair_json
from .readers_base import BaseParser

logger = logging.getLogger(__name__)

class JSONParser(BaseParser):
    """JSON 文件解析器"""
    
    def parse(self, file_path: str) -> Optional[str]:
        """解析 JSON 文件并返回格式化文本"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 尝试标准 JSON 解析
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # 尝试修复损坏的 JSON
                data = repair_json(content, ensure_ascii=False)
                if data is None:
                    return None
            
            # 添加文件标识
            # result = [f"=== JSON文件: {Path(file_path).name} ===\n"]
            # result.append("JSON内容:")
            # result.append(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False))

            return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False)
            
        except Exception as e:
            return f"解析JSON文件失败: {str(e)}"
    
# 注册解析器
# ParserFactory.register_parser(JSONParser)

if __name__ == "__main__":
    # 测试示例
    parser = JSONParser()
    
    # 创建测试JSON文件
    test_data = {
        "name": "测试项目",
        "version": "1.0.0",
        "description": "这是一个测试JSON文件",
        "config": {
            "database": {
                "host": "localhost",
                "port": 3306
            },
            "features": ["feature1", "feature2", "feature3"]
        },
        "users": [
            {"id": 1, "name": "用户1", "email": "user1@example.com"},
            {"id": 2, "name": "用户2", "email": "user2@example.com"}
        ]
    }
    
    test_file = "test.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    # 测试解析
    result = parser.parse(test_file)
    if result:
        logger.info("解析结果:")
        logger.info(result)
    
    # 清理测试文件
    try:
        Path(test_file).unlink()
    except Exception:
        pass
