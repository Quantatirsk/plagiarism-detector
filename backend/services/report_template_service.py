"""
报告模板服务 - 管理报告生成的提示词模板和配置
"""
from typing import Any

from backend.models.report_models import ReportTemplate, ReportType
from backend.services.base_service import BaseService, singleton


@singleton
class ReportTemplateService(BaseService):
    """报告模板服务 - 管理各类报告的生成模板"""

    def _initialize(self):
        """初始化模板数据"""
        self.templates: dict[str, ReportTemplate] = {}
        self._load_templates()

    def _load_templates(self):
        """加载所有报告模板"""
        # 文档报告模板
        self.templates["document_zh"] = ReportTemplate(
            type=ReportType.DOCUMENT,
            language="zh",
            system_prompt="""你是招采合规分析师，专注于识别投标文件雷同和串标风险。

核心任务：基于数据生成简洁、精准的风险分析报告。

输出要求：
1. **简洁为先**：总字数控制在 800-1200 字以内
2. **数据驱动**：直接引用数据支撑结论，避免空泛描述
3. **聚焦关键**：只分析高风险点和核心发现，不重复罗列已有数据
4. **结论明确**：风险等级、主要问题、建议行动要清晰可执行
5. **招采话术**：使用"投标人"、"评标"、"串标迹象"等行业术语""",
            user_prompt_template="""## 任务：生成投标文件雷同性分析报告

### 文档基础信息
- 文件名称：{document_title}
- 综合相似度：{total_similarity_score:.1%}
- 风险等级：{risk_level}
- 对比来源数量：{sources_count}

### 相似度来源数据
{sources_analysis}

### 重点雷同片段
{match_details}

### 统计数据
{statistics}

---

### 报告结构（总字数 ≤ 1200 字）

**1. 风险概述**（200字以内）
- 一句话总结风险水平和核心问题
- 关键数据点：相似度、主要来源、高风险片段数量

**2. 关键发现**（400字以内）
- 列举 TOP 3 高相似度来源，说明风险含义
- 分析重点雷同片段的特征（技术方案/报价/资质等）
- 标注最具风险的内容类型

**3. 合规建议**（300字以内）
- 针对高风险点的具体处理建议（要求澄清、现场核查、专家评审等）
- 标注需要重点关注的对比文件

---

**输出规范**：
- 直接陈述结论，不要重复原始数据
- 每个发现必须引用具体数据支撑
- 建议要具体可执行，避免空话""",
            sections=["executive_summary", "risk_assessment", "source_analysis", "key_matches", "recommendations"],
            chart_configs={
                "similarity_pie": {"type": "pie", "title": "相似度来源分布"},
                "risk_gauge": {"type": "gauge", "title": "风险评估", "max": 100}
            }
        )

        self.templates["document_en"] = ReportTemplate(
            type=ReportType.DOCUMENT,
            language="en",
            system_prompt="""You are a procurement compliance analyst specializing in tender collusion and bid-rigging detection.

Core Task: Generate concise, data-driven risk analysis reports.

Output Requirements:
1. **Brevity First**: Total word count 800-1200 words
2. **Data-Driven**: Support conclusions with specific data, avoid generic statements
3. **Focus on Key Issues**: Analyze only high-risk findings, don't repeat raw data
4. **Clear Conclusions**: Risk level, main issues, and actionable recommendations
5. **Industry Language**: Use "bidder", "evaluation", "collusion indicators" terminology""",
            user_prompt_template="""## Task: Generate Tender Similarity Analysis Report

### Document Information
- Document Title: {document_title}
- Overall Similarity: {total_similarity_score:.1%}
- Risk Level: {risk_level}
- Comparison Sources: {sources_count}

### Similarity Source Data
{sources_analysis}

### Critical Similar Segments
{match_details}

### Statistical Data
{statistics}

---

### Report Structure (Total ≤ 1200 words)

**1. Risk Overview** (≤200 words)
- One-sentence summary of risk level and core issues
- Key data points: similarity score, main sources, high-risk segment count

**2. Key Findings** (≤400 words)
- List TOP 3 high-similarity sources with risk implications
- Analyze characteristics of critical segments (technical/pricing/qualification)
- Identify most risky content types

**3. Compliance Recommendations** (≤300 words)
- Specific actions for high-risk items (clarification, site verification, expert review)
- Flag comparison documents requiring special attention

---

**Output Standards**:
- State conclusions directly, don't repeat raw data
- Every finding must cite specific supporting data
- Recommendations must be specific and actionable""",
            sections=["executive_summary", "risk_assessment", "source_analysis", "key_matches", "recommendations"],
            chart_configs={
                "similarity_pie": {"type": "pie", "title": "Similarity Source Distribution"},
                "risk_gauge": {"type": "gauge", "title": "Risk Assessment", "max": 100}
            }
        )

        # 对比报告模板
        self.templates["comparison_zh"] = ReportTemplate(
            type=ReportType.COMPARISON,
            language="zh",
            system_prompt="""你是招采对比分析专家，专注识别投标文件间的串标迹象和协同编制风险。

核心任务：基于双向相似度数据，快速判断两份标书的关联关系。

输出要求：
1. **精准简洁**：总字数控制在 600-1000 字以内
2. **结论优先**：开篇直接给出串标风险判断
3. **数据说话**：用双向相似度、独有内容占比等数据支撑结论
4. **聚焦异常**：重点分析不对称相似度和高度雷同片段
5. **可执行建议**：明确告知是否需要进一步调查或处理""",
            user_prompt_template="""## 任务：生成双标对比分析报告

### 对比文档
- 文档 A：{document_a_title}
- 文档 B：{document_b_title}

### 相似度指标
- A→B 雷同度：{similarity_a_to_b:.1%}
- B→A 雷同度：{similarity_b_to_a:.1%}
- 共同内容占比：{common_similarity:.1%}
- A 独有内容：{unique_a_ratio:.1%}
- B 独有内容：{unique_b_ratio:.1%}

### 重点雷同片段
{match_details}

### 并排对照数据
{side_by_side_analysis}

---

### 报告结构（总字数 ≤ 1000 字）

**1. 关联关系判断**（150字以内）
- 一句话结论：是否存在串标或协同编制迹象
- 关键依据：相似度数据异常情况

**2. 核心发现**（400字以内）
- 分析双向相似度差异（如 A→B 高但 B→A 低，说明可能存在抄袭方向）
- 标注高度雷同片段的位置和内容类型
- 评估独有内容占比的合理性

**3. 风险评估与建议**（200字以内）
- 明确风险等级（低/中/高/严重）
- 给出具体处理建议（约谈澄清、要求说明、废标等）

---

**输出规范**：
- 避免重复罗列原始数据
- 重点解释数据背后的风险含义
- 建议要有可操作性""",
            sections=["comparison_summary", "similarity_matrix", "content_distribution", "key_matches", "pattern_analysis", "conclusions"],
            chart_configs={
                "similarity_matrix": {"type": "heatmap", "title": "相似度矩阵"},
                "content_distribution": {"type": "stacked_bar", "title": "内容分布"}
            }
        )

        # 项目报告模板
        self.templates["project_zh"] = ReportTemplate(
            type=ReportType.PROJECT,
            language="zh",
            system_prompt="""你是招采合规顾问，负责项目级雷同风险全局分析。

核心任务：从项目整体视角，快速识别系统性风险和重点监控对象。

输出要求：
1. **宏观聚焦**：总字数控制在 1000-1500 字以内
2. **关键优先**：突出高风险文件和异常模式，忽略正常范围内的数据
3. **数据洞察**：从统计分布中提炼规律，不要简单复述数字
4. **风险分层**：明确哪些是重点风险、哪些是一般关注
5. **决策导向**：告知项目业主或监管方应采取什么行动""",
            user_prompt_template="""## 任务：生成项目级雷同风险报告

### 项目概况
- 项目名称：{project_name}
- 投标文件总数：{total_documents}
- 比对总数：{total_comparisons}
- 平均雷同度：{average_similarity:.1%}
- 高风险文件数：{high_risk_count}

### 统计分析
{statistics_analysis}

### 雷同度分布
{similarity_distribution}

### 高风险文件
{high_risk_documents}

### 异常检测
{anomalies}

### 相似网络
{network_analysis}

---

### 报告结构（总字数 ≤ 1500 字）

**1. 项目风险总览**（250字以内）
- 一句话总结项目整体风险水平
- 关键指标：高风险文件占比、平均雷同度、异常数量

**2. 重点风险标的**（500字以内）
- 列举 TOP 5 高风险文件，说明具体风险点
- 标注是否存在集中雷同或关联串标迹象
- 引用具体相似度数据和异常特征

**3. 模式与趋势**（350字以内）
- 分析雷同度分布特征（是否存在集中高峰、异常离群值）
- 识别可能的协同编制或模板复用模式
- 评估相似网络中的核心节点

**4. 处理建议**（250字以内）
- 明确哪些文件需要立即审查或废标
- 哪些需要进一步调查或约谈
- 项目层面的改进措施（加强资格预审、引入第三方核查等）

---

**输出规范**：
- 避免逐条罗列统计数据，提炼关键洞察
- 重点文件要有明确名称和风险评级
- 建议要分轻重缓急""",
            sections=["project_summary", "overall_assessment", "statistical_analysis", "high_risk_cases", "behavior_patterns", "anomaly_detection", "recommendations"],
            chart_configs={
                "similarity_distribution": {"type": "histogram", "title": "相似度分布"},
                "network_graph": {"type": "network", "title": "文档关系网络"},
                "trend_analysis": {"type": "line", "title": "趋势分析"}
            }
        )

    def get_template(self, report_type: ReportType, language: str = "zh") -> ReportTemplate:
        """获取报告模板"""
        self._ensure_initialized()
        template_key = f"{report_type.value}_{language}"
        if template_key not in self.templates:
            # 如果没有指定语言的模板，使用中文作为默认
            template_key = f"{report_type.value}_zh"

        template = self.templates.get(template_key)
        if not template:
            raise ValueError(f"Template not found for type: {report_type}, language: {language}")

        return template

    def format_user_prompt(self, template: ReportTemplate, data: dict[str, Any]) -> str:
        """格式化用户提示词"""
        try:
            return template.user_prompt_template.format(**data)
        except KeyError as e:
            self.logger.error(f"Missing template variable: {e}")
            raise ValueError(f"Template formatting failed, missing variable: {e}")

    def get_available_templates(self) -> list[dict[str, str]]:
        """获取可用模板列表"""
        self._ensure_initialized()  # 确保初始化
        return [
            {
                "type": template.type.value,
                "language": template.language,
                "sections_count": str(len(template.sections))
            }
            for template in self.templates.values()
        ]

    def validate_template_data(self, template: ReportTemplate, data: dict[str, Any]) -> bool:
        """验证模板数据的完整性"""
        import re

        # 提取模板中需要的变量
        variables = re.findall(r'\{(\w+)\}', template.user_prompt_template)

        missing_vars = [var for var in variables if var not in data]
        if missing_vars:
            self.logger.warning(f"Missing template variables: {missing_vars}")
            return False

        return True

    def get_chart_configs(self, report_type: ReportType, language: str = "zh") -> dict[str, Any]:
        """获取图表配置"""
        template = self.get_template(report_type, language)
        return template.chart_configs

    def customize_template(
        self,
        report_type: ReportType,
        language: str,
        custom_sections: list[str] | None = None,
        custom_prompts: dict[str, str] | None = None
    ) -> ReportTemplate:
        """自定义报告模板"""
        base_template = self.get_template(report_type, language)

        # 创建自定义模板副本
        custom_template = ReportTemplate(
            type=base_template.type,
            language=base_template.language,
            system_prompt=custom_prompts.get("system_prompt", base_template.system_prompt) if custom_prompts else base_template.system_prompt,
            user_prompt_template=custom_prompts.get("user_prompt_template", base_template.user_prompt_template) if custom_prompts else base_template.user_prompt_template,
            sections=custom_sections if custom_sections else base_template.sections,
            chart_configs=base_template.chart_configs
        )

        return custom_template
