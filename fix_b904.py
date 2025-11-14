"""批量修复 B904 异常处理问题"""
import json
import re
import subprocess
from pathlib import Path

# 获取所有 B904 错误
result = subprocess.run(
    ["ruff", "check", "backend", "--select", "B904", "--output-format=json"],
    capture_output=True,
    text=True,
    cwd=Path(__file__).parent
)

errors = json.loads(result.stdout)
print(f"Found {len(errors)} B904 errors")

# 按文件分组
files_to_fix = {}
for error in errors:
    filepath = error["filename"]
    line_no = error["location"]["row"]

    if filepath not in files_to_fix:
        files_to_fix[filepath] = []
    files_to_fix[filepath].append(line_no)

# 修复每个文件
for filepath, line_numbers in files_to_fix.items():
    print(f"\nFixing {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 从后往前处理，避免行号变化
    for line_no in sorted(line_numbers, reverse=True):
        idx = line_no - 1
        line = lines[idx]

        # 匹配 raise 语句
        match = re.match(r'^(\s+)raise\s+(.+?)(\s*)$', line)
        if match:
            indent, raise_content, trailing = match.groups()

            # 检查上一行是否有 except
            if idx > 0:
                except_line = lines[idx - 1]
                # 查找异常变量名
                except_match = re.search(r'except\s+\w+\s+as\s+(\w+):', except_line)
                if except_match:
                    var_name = except_match.group(1)
                    # 添加 from
                    if ' from ' not in line:
                        new_line = f"{indent}raise {raise_content.rstrip()} from {var_name}{trailing}"
                        lines[idx] = new_line
                        print(f"  Line {line_no}: Added 'from {var_name}'")
                else:
                    # 没有变量名，可能是其他情况，跳过或使用 None
                    if 'except Exception' in except_line or 'except:' in except_line:
                        # 查找更早的 except 行
                        for prev_idx in range(idx - 1, max(0, idx - 10), -1):
                            prev_line = lines[prev_idx]
                            if re.search(r'except\s+\w+\s+as\s+(\w+):', prev_line):
                                var_match = re.search(r'as\s+(\w+)', prev_line)
                                if var_match:
                                    var_name = var_match.group(1)
                                    if ' from ' not in line:
                                        new_line = f"{indent}raise {raise_content.rstrip()} from {var_name}{trailing}"
                                        lines[idx] = new_line
                                        print(f"  Line {line_no}: Added 'from {var_name}'")
                                    break

    # 写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

print("\nDone!")
