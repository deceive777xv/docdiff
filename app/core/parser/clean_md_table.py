import re
from typing import List, Optional

def clean_md_table_cells(
    md_text: str,
    nan_patterns: Optional[List[str]] = None,
    remove_patterns: Optional[List[str]] = None,
) -> str:
    """
    清理 Markdown 表格中的单元格内容：
    - 将 NaN 类占位符替换为空字符串
    - 删除无用的图片公式（如 =DISPIMG(...)）

    Args:
        md_text: 包含 Markdown 表格的文本
        nan_patterns: 需要替换为空的精确匹配字符串列表，默认 ['NaN', 'nan', 'None', 'NA', 'N/A']
        remove_patterns: 需要删除的正则表达式模式列表（删除匹配到的子串），
                         默认 [r'=DISPIMG\([^)]*\)'] 匹配 =DISPIMG(...)

    Returns:
        清理后的文本
    """
    if nan_patterns is None:
        nan_patterns = ['NaN', 'nan', 'None', 'NA', 'N/A']
    if remove_patterns is None:
        remove_patterns = [r'=DISPIMG\([^)]*\)']

    # 预编译所有删除用的正则
    remove_regexes = [re.compile(pattern, re.IGNORECASE) for pattern in remove_patterns]

    lines = md_text.split('\n')
    result_lines = []
    table_separator_pattern = re.compile(r'^\s*\|[\s\-:|]+\|\s*$')

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|') and stripped.endswith('|'):
            # 表格分隔行（|---|---|）原样保留
            if table_separator_pattern.match(stripped):
                result_lines.append(line)
                continue

            # 处理表格数据行
            cells = line.split('|')
            processed_cells = []

            for i, cell in enumerate(cells):
                if i == 0 or i == len(cells) - 1:
                    processed_cells.append(cell)  # 保留首尾的空串
                else:
                    raw = cell
                    content = raw.strip()

                    # 1. 删除指定模式（如 =DISPIMG(...)）
                    for regex in remove_regexes:
                        content = regex.sub('', content)

                    # 2. 替换 NaN 类占位符（精确匹配）
                    for nan in nan_patterns:
                        if content == nan:
                            content = ''
                            break

                    # 3. 保留原始单元格的空格风格（简单还原）
                    # 如果原本有前导或尾随空格，尽量保持格式整齐
                    if raw.startswith(' ') and raw.endswith(' '):
                        processed_cells.append(f' {content} ')
                    elif raw.startswith(' '):
                        processed_cells.append(f' {content}')
                    elif raw.endswith(' '):
                        processed_cells.append(f'{content} ')
                    else:
                        processed_cells.append(content)

            new_line = '|'.join(processed_cells)
            result_lines.append(new_line)
        else:
            # 非表格行，直接保留
            result_lines.append(line)

    return '\n'.join(result_lines)

