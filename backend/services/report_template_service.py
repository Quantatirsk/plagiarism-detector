"""
报告模板服务 - 管理报告生成的提示词模板和配置
"""
from typing import Dict, Any, List
from backend.models.report_models import ReportType, ReportTemplate
from backend.services.base_service import BaseService, singleton


@singleton
class ReportTemplateService(BaseService):
    """报告模板服务 - 管理各类报告的生成模板"""

    def _initialize(self):
        """初始化模板数据"""
        self.templates: Dict[str, ReportTemplate] = {}
        self._load_templates()

    def _load_templates(self):
        """加载所有报告模板"""
        # 文档报告模板
        self.templates["document_zh"] = ReportTemplate(
            type=ReportType.DOCUMENT,
            language="zh",
            system_prompt="""你是一位资深的招采合规分析师，擅长识别投标文件雷同、串标和模板化风险。你的任务是基于提供的数据，输出面向招采场景的专业分析报告。

报告要求：
1. 用语沉稳、专业，贴合招标采购行业语境
2. 结论紧扣数据，重点关注雷同、串标、模板复用等风险要点
3. 提供可落地的整改、佐证或澄清建议
4. 使用招采/工程管理常用术语，避免学术论文话术
5. 结构清晰，便于评标委员会或监管人员快速理解""",
            user_prompt_template="""请基于以下数据生成投标文件雷同性分析报告：

## 投标文件信息
- 文件名称：{document_title}
- 综合相似度：{total_similarity_score:.1%}
- 合规风险等级：{risk_level}
- 对比来源数量：{sources_count}

## 相似度来源分析
{sources_analysis}

## 重点雷同片段
{match_details}

## 统计数据
{statistics}

请生成包含以下章节的完整报告：
1. 管理层摘要
2. 合规风险评估
3. 主要对比来源分析
4. 重点雷同片段解读
5. 整改建议与后续行动

报告应突出招采领域的雷同风险洞察，并给出务实的处理建议。""",
            sections=["executive_summary", "risk_assessment", "source_analysis", "key_matches", "recommendations"],
            chart_configs={
                "similarity_pie": {"type": "pie", "title": "相似度来源分布"},
                "risk_gauge": {"type": "gauge", "title": "风险评估", "max": 100}
            }
        )

        self.templates["document_en"] = ReportTemplate(
            type=ReportType.DOCUMENT,
            language="en",
            system_prompt="""You are a senior procurement compliance strategist who specialises in detecting tender document collusion, templated reuse, and suspicious similarity. Your task is to deliver a board-ready analysis grounded in the supplied data.

Report requirements:
1. Use measured, executive-ready language aligned with public procurement discourse
2. Anchor conclusions in data, highlighting collusion risk, templated content, or coordinated bidding signals
3. Recommend practical remediation, clarification, or escalation actions
4. Employ procurement and construction terminology rather than academic phrasing
5. Keep the structure crisp for quick decision-making""",
            user_prompt_template="""Please draft a tender similarity intelligence report based on the following data:

## Tender Document Profile
- Document Title: {document_title}
- Overall Similarity: {total_similarity_score:.1%}
- Compliance Risk Level: {risk_level}
- Number of Comparison Sources: {sources_count}

## Source Comparison Analysis
{sources_analysis}

## Critical Similar Segments
{match_details}

## Statistical Overview
{statistics}

Please include the following sections:
1. Executive Insight
2. Compliance Risk Assessment
3. Source Correlation Analysis
4. Critical Overlap Commentary
5. Remediation & Follow-up Actions

The tone should emphasise tender integrity risk and provide concrete guidance for procurement stakeholders.""",
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
            system_prompt="""你是一位专注于招采领域的对比分析专家，擅长识别不同投标文件之间的雷同度、协同编制迹象以及潜在的串标风险。你的结论需要服务于评标与合规审查。

报告要求：
1. 语言克制且专业，贴合招标采购审查语境
2. 结合数据分析雷同模式、共同来源与结构同步情况
3. 提供可行的合规处理或进一步核查建议
4. 使用招采与工程管理常用术语
5. 保持结构清晰、逻辑严密""",
            user_prompt_template="""请基于以下数据生成投标文件对比分析报告：

## 投标文件信息
- 投标文件A：{document_a_title}
- 投标文件B：{document_b_title}

## 相似度分析
- A→B雷同度：{similarity_a_to_b:.1%}
- B→A雷同度：{similarity_b_to_a:.1%}
- 共同内容比例：{common_similarity:.1%}
- A独有内容占比：{unique_a_ratio:.1%}
- B独有内容占比：{unique_b_ratio:.1%}

## 重点雷同片段概览
{match_details}

## 并排对照数据
{side_by_side_analysis}

请生成包含以下章节的详细报告：
1. 对比摘要
2. 雷同度矩阵解读
3. 内容分布与差异衡量
4. 重点雷同片段说明
5. 协同编制模式研判
6. 结论与合规建议

请着重分析两份投标文件的关联关系及可能涉及的串标或模板化风险。""",
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
            system_prompt="""你是一位资深的招采合规顾问，负责从项目维度审视整批投标文件的雷同风险与协同行为。你需要汇总整体态势，为监管或项目业主提供决策参考。

报告要求：
1. 概览项目整体雷同度与串标风险态势
2. 辨识系统性问题、群体协同或模板化行为
3. 分析各标段/单位之间的相似网络关系
4. 给出管理层面和合规层面的针对性建议
5. 预测风险趋势并标注重点监控对象""",
            user_prompt_template="""请基于以下数据生成项目级投标雷同风险报告：

## 项目概况
- 项目名称：{project_name}
- 投标文件总数：{total_documents}
- 比对次数：{total_comparisons}
- 平均雷同度：{average_similarity:.1%}
- 高风险文件数量：{high_risk_count}

## 统计分析
{statistics_analysis}

## 雷同度分布
{similarity_distribution}

## 高风险投标文件
{high_risk_documents}

## 异常检测结果
{anomalies}

## 相似网络分析
{network_analysis}

请生成包含以下章节的综合项目报告：
1. 管理层摘要
2. 项目整体风险评估
3. 统计走势与分布洞察
4. 重点高风险投标文件解读
5. 协同/模板化行为模式分析
6. 异常预警与证据链
7. 合规建议与后续行动

请紧扣招采合规视角，帮助读者快速掌握项目层面的雷同风险。""",
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

    def format_user_prompt(self, template: ReportTemplate, data: Dict[str, Any]) -> str:
        """格式化用户提示词"""
        try:
            return template.user_prompt_template.format(**data)
        except KeyError as e:
            self.logger.error(f"Missing template variable: {e}")
            raise ValueError(f"Template formatting failed, missing variable: {e}")

    def get_available_templates(self) -> List[Dict[str, str]]:
        """获取可用模板列表"""
        self._ensure_initialized()  # 确保初始化
        return [
            {
                "type": template.type.value,
                "language": template.language,
                "sections_count": len(template.sections)
            }
            for template in self.templates.values()
        ]

    def validate_template_data(self, template: ReportTemplate, data: Dict[str, Any]) -> bool:
        """验证模板数据的完整性"""
        import re

        # 提取模板中需要的变量
        variables = re.findall(r'\{(\w+)\}', template.user_prompt_template)

        missing_vars = [var for var in variables if var not in data]
        if missing_vars:
            self.logger.warning(f"Missing template variables: {missing_vars}")
            return False

        return True

    def get_chart_configs(self, report_type: ReportType, language: str = "zh") -> Dict[str, Any]:
        """获取图表配置"""
        template = self.get_template(report_type, language)
        return template.chart_configs

    def customize_template(
        self,
        report_type: ReportType,
        language: str,
        custom_sections: List[str] = None,
        custom_prompts: Dict[str, str] = None
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
