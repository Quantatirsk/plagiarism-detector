"""
报告生成服务 - 整合LLM、模板和数据处理，生成完整的抄袭检测报告
"""
import html
import json
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import TYPE_CHECKING, Any

from backend.models.report_models import (
    ComparisonReportData,
    DocumentReportData,
    GeneratedReport,
    MatchDetail,
    ProjectReportData,
    ProjectStatistics,
    ReportGenerationRequest,
    ReportProgress,
    ReportTemplate,
    ReportType,
)
from backend.services.base_service import BaseService, singleton
from backend.services.report_data_processor import ReportDataProcessor
from backend.services.report_template_service import ReportTemplateService
from backend.services.service_factory import ServiceFactory

if TYPE_CHECKING:
    from backend.services.llm_service import LLMService


@singleton
class ReportGeneratorService(BaseService):
    """报告生成服务 - 核心报告生成引擎"""

    SIMILARITY_SEGMENT_THRESHOLD: float = 0.8
    LARGE_MATCH_DETAIL_LIMIT: int = 1000

    def _initialize(self):
        """初始化服务依赖"""
        self.llm_service: LLMService = ServiceFactory.get_llm_service()
        self.template_service: ReportTemplateService = ReportTemplateService()
        self.data_processor: ReportDataProcessor = ReportDataProcessor()
        self.active_generations: dict[str, ReportProgress] = {}

    async def generate_report(
        self,
        request: ReportGenerationRequest,
        stream: bool = False
    ) -> GeneratedReport | AsyncGenerator[str, None]:
        """
        生成报告 - 主入口方法

        Args:
            request: 报告生成请求
            stream: 是否流式生成

        Returns:
            完整报告或流式生成器
        """
        self._ensure_initialized()

        # 生成任务ID和进度跟踪
        task_id = str(uuid.uuid4())
        self.active_generations[task_id] = ReportProgress(
            task_id=task_id,
            progress=0.0,
            stage="initializing",
            message="初始化报告生成..."
        )

        try:
            if stream:
                return self._generate_report_stream(task_id, request)
            else:
                return await self._generate_report_complete(task_id, request)

        except Exception as e:
            self.logger.error(f"Report generation failed: {e}", task_id=task_id)
            self.active_generations[task_id].error = str(e)
            raise
        finally:
            # 清理完成的任务
            if task_id in self.active_generations:
                del self.active_generations[task_id]

    async def _generate_report_complete(
        self,
        task_id: str,
        request: ReportGenerationRequest
    ) -> GeneratedReport:
        """生成完整报告"""
        try:
            # 阶段1: 数据收集和处理
            self._update_progress(task_id, 0.1, "data_collection", "收集和处理数据...")
            report_data = await self._collect_report_data(request)

            # 阶段2: 准备模板和提示词
            self._update_progress(task_id, 0.3, "template_preparation", "准备报告模板...")
            template = self.template_service.get_template(request.type, request.language)
            formatted_data = self._prepare_template_data(report_data, request)

            # 验证模板数据
            if not self.template_service.validate_template_data(template, formatted_data):
                raise ValueError("Template data validation failed")

            user_prompt = self.template_service.format_user_prompt(template, formatted_data)

            # 阶段3: LLM生成报告内容
            self._update_progress(task_id, 0.5, "llm_generation", "生成报告内容...")
            messages = [
                {"role": "system", "content": template.system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # 根据报告类型调整 token 限制以控制输出长度
            max_tokens_map = {
                "document": 2000,    # 800-1200 字
                "comparison": 1500,  # 600-1000 字
                "project": 2500      # 1000-1500 字
            }
            max_tokens = max_tokens_map.get(request.type.value, 2000)

            llm_response = await self.llm_service.chat_completion(
                messages=messages,
                model=request.llm_model,
                temperature=0.3,  # 降低温度以获得更精准、简洁的输出
                max_tokens=max_tokens
            )

            # 阶段4: 处理和结构化报告内容
            self._update_progress(task_id, 0.8, "content_processing", "处理报告内容...")
            llm_payload = self._process_llm_response(llm_response, template)
            report_content = self._build_structured_content(request, report_data, llm_payload)

            # 阶段5: 生成最终报告
            self._update_progress(task_id, 0.9, "report_finalization", "完成报告生成...")
            generated_report = GeneratedReport(
                id=task_id,
                type=request.type,
                title=self._generate_report_title(request, report_data),
                summary=self._extract_summary(report_content),  # 前端已不再使用，保留仅为数据完整性
                content=report_content,
                data=report_data,
                generated_at=datetime.utcnow(),
                language=request.language
            )

            self._update_progress(task_id, 1.0, "completed", "报告生成完成")
            return generated_report

        except Exception as e:
            self._update_progress(task_id, -1, "error", f"生成失败: {e!s}")
            raise

    async def _generate_report_stream(
        self,
        task_id: str,
        request: ReportGenerationRequest
    ) -> AsyncGenerator[str, None]:
        """流式生成报告"""
        try:
            # 数据准备阶段
            yield json.dumps({"type": "progress", "stage": "data_collection", "progress": 0.1}) + "\n"

            report_data = await self._collect_report_data(request)
            template = self.template_service.get_template(request.type, request.language)
            formatted_data = self._prepare_template_data(report_data, request)
            user_prompt = self.template_service.format_user_prompt(template, formatted_data)

            yield json.dumps({"type": "progress", "stage": "llm_generation", "progress": 0.3}) + "\n"

            # 流式生成
            messages = [
                {"role": "system", "content": template.system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # 根据报告类型调整 token 限制以控制输出长度
            max_tokens_map = {
                "document": 2000,    # 800-1200 字
                "comparison": 1500,  # 600-1000 字
                "project": 2500      # 1000-1500 字
            }
            max_tokens = max_tokens_map.get(request.type.value, 2000)

            content_buffer = ""
            async for chunk in self.llm_service.stream_chat_completion(
                messages=messages,
                model=request.llm_model,
                temperature=0.3,  # 降低温度以获得更精准、简洁的输出
                max_tokens=max_tokens
            ):
                content_buffer += chunk
                yield json.dumps({"type": "content", "chunk": chunk}) + "\n"

            llm_payload = {
                "full_content": content_buffer,
                "sections": {"content": content_buffer},
                "generated_at": datetime.utcnow().isoformat(),
                "model_used": request.llm_model or getattr(self.llm_service, "default_model", None)
            }
            report_content = self._build_structured_content(request, report_data, llm_payload)

            generated_report = GeneratedReport(
                id=task_id,
                type=request.type,
                title=self._generate_report_title(request, report_data),
                summary=self._extract_summary(report_content),  # 前端已不再使用，保留仅为数据完整性
                content=report_content,
                data=report_data,
                generated_at=datetime.utcnow(),
                language=request.language
            )

            report_payload = json.loads(generated_report.json())
            yield json.dumps({"type": "completed", "report": report_payload}) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    async def _collect_report_data(self, request: ReportGenerationRequest):
        """收集报告数据"""
        if request.type == ReportType.DOCUMENT:
            if not request.document_id:
                raise ValueError("Document ID is required for document report")
            return await self.data_processor.process_document_report_data(
                request.document_id,
                request.max_matches_detail
            )

        elif request.type == ReportType.COMPARISON:
            if not request.document_a_id or not request.document_b_id:
                raise ValueError("Both document IDs are required for comparison report")
            return await self.data_processor.process_comparison_report_data(
                request.document_a_id,
                request.document_b_id
            )

        elif request.type == ReportType.PROJECT:
            if not request.project_id:
                raise ValueError("Project ID is required for project report")
            return await self.data_processor.process_project_report_data(
                request.project_id,
                request.include_network_graph
            )

        else:
            raise ValueError(f"Unsupported report type: {request.type}")

    def _prepare_template_data(
        self,
        report_data,
        request: ReportGenerationRequest
    ) -> dict[str, Any]:
        """准备模板数据"""
        if request.type == ReportType.DOCUMENT:
            return self._prepare_document_template_data(report_data)
        elif request.type == ReportType.COMPARISON:
            return self._prepare_comparison_template_data(report_data)
        elif request.type == ReportType.PROJECT:
            return self._prepare_project_template_data(report_data)
        else:
            raise ValueError(f"Unsupported report type: {request.type}")

    def _prepare_document_template_data(self, data: DocumentReportData) -> dict[str, Any]:
        """准备文档报告模板数据"""
        # 格式化来源分析
        sources_analysis = self._format_sources_analysis(data.sources)

        # 格式化匹配详情
        match_details = self._format_match_details(data.top_matches)

        # 格式化统计数据
        statistics = self._format_statistics(data.statistics)

        return {
            'document_title': data.document_title,
            'total_similarity_score': data.total_similarity_score,
            'risk_level': data.risk_level.value,
            'sources_count': len(data.sources),
            'sources_analysis': sources_analysis,
            'match_details': match_details,
            'statistics': statistics
        }

    def _prepare_comparison_template_data(self, data: ComparisonReportData) -> dict[str, Any]:
        """准备对比报告模板数据"""
        match_details = self._format_match_details(data.match_details)
        total_matches = len(data.match_details)
        overflow_warning = None
        if total_matches > self.LARGE_MATCH_DETAIL_LIMIT:
            overflow_warning = (
                f"⚠️ 匹配结果包含 {total_matches} 个片段，已仅抽取关键高风险片段供分析。"
            )
            match_details = f"{overflow_warning}\n\n{match_details}" if match_details else overflow_warning
        side_by_side_analysis = self._format_side_by_side_analysis(data.side_by_side_sections)

        return {
            'document_a_title': data.document_a_title,
            'document_b_title': data.document_b_title,
            'similarity_a_to_b': data.similarity_a_to_b,
            'similarity_b_to_a': data.similarity_b_to_a,
            'common_similarity': data.common_similarity,
            'unique_a_ratio': data.unique_a_ratio,
            'unique_b_ratio': data.unique_b_ratio,
            'total_match_details': total_matches,
            'large_dataset_warning': overflow_warning,
            'match_details': match_details,
            'side_by_side_analysis': side_by_side_analysis
        }

    def _prepare_project_template_data(self, data: ProjectReportData) -> dict[str, Any]:
        """准备项目报告模板数据"""
        return {
            'project_name': data.project_name,
            'total_documents': data.statistics.total_documents,
            'total_comparisons': data.statistics.total_comparisons,
            'average_similarity': data.statistics.average_similarity,
            'high_risk_count': len(data.high_risk_documents),
            'statistics_analysis': self._format_project_statistics(data.statistics),
            'similarity_distribution': json.dumps(data.statistics.similarity_distribution, ensure_ascii=False),
            'high_risk_documents': self._format_high_risk_documents(data.high_risk_documents),
            'anomalies': json.dumps(data.anomalies, ensure_ascii=False),
            'network_analysis': json.dumps(data.similarity_network, ensure_ascii=False)
        }

    def _format_sources_analysis(self, sources) -> str:
        """格式化来源分析"""
        if not sources:
            return "未发现明显的相似来源。"

        lines = []
        for i, source in enumerate(sources[:10], 1):  # 显示前10个来源
            lines.append(f"{i}. {source.document_title} - 相似度: {source.similarity_score:.1%}, 匹配数: {source.match_count}")

        return "\n".join(lines)

    def _format_match_details(self, matches: list[MatchDetail]) -> str:
        """格式化匹配详情"""
        if not matches:
            return "无高风险匹配内容。"

        lines = []
        for i, match in enumerate(matches[:5], 1):  # 显示前5个匹配
            lines.append(f"{i}. 相似度: {match.similarity_score:.1%}")
            lines.append(f"   原文: {match.source_text[:100]}...")
            lines.append(f"   对比: {match.target_text[:100]}...")
            lines.append("")

        return "\n".join(lines)

    def _format_statistics(self, stats: dict[str, Any]) -> str:
        """格式化统计数据"""
        return f"""
总来源数: {stats.get('total_sources', 0)}
平均相似度: {stats.get('average_similarity', 0):.1%}
最高相似度: {stats.get('max_similarity', 0):.1%}
高风险来源: {stats.get('high_risk_sources', 0)}
"""

    def _format_side_by_side_analysis(self, sections: list[dict[str, Any]]) -> str:
        """格式化并排分析"""
        if not sections:
            return "无并排对比数据。"

        lines = [f"发现 {len(sections)} 个相似段落："]
        for i, section in enumerate(sections[:3], 1):  # 显示前3个段落
            lines.append(f"\n段落 {i} (相似度: {section.get('similarity', 0):.1%}):")
            lines.append(f"文档A: {section.get('text_a', '')[:80]}...")
            lines.append(f"文档B: {section.get('text_b', '')[:80]}...")

        return "\n".join(lines)

    def _format_project_statistics(self, stats: ProjectStatistics) -> str:
        """格式化项目统计"""
        return f"""
文档总数: {stats.total_documents}
比较总数: {stats.total_comparisons}
平均相似度: {stats.average_similarity:.1%}
高风险文档: {stats.high_risk_count}
"""

    def _format_high_risk_documents(self, high_risk_docs: list[DocumentReportData]) -> str:
        """格式化高风险文档"""
        if not high_risk_docs:
            return "无高风险文档。"

        lines = []
        for doc in high_risk_docs[:5]:  # 显示前5个
            lines.append(f"• {doc.document_title} - 风险等级: {doc.risk_level.value}, 相似度: {doc.total_similarity_score:.1%}")

        return "\n".join(lines)

    def _process_llm_response(self, response: Any, template: ReportTemplate) -> dict[str, Any]:
        """处理LLM响应"""
        content = response.choices[0].message.content

        # 尝试按章节分割内容
        sections = {}
        current_section = "content"
        sections[current_section] = content

        # 如果模板定义了章节，尝试分割
        for section_name in template.sections:
            # 简单的章节提取逻辑
            if section_name in content.lower():
                sections[section_name] = self._extract_section_content(content, section_name)

        return {
            "full_content": content,
            "sections": sections,
            "generated_at": datetime.utcnow().isoformat(),
            "model_used": response.model
        }

    def _extract_section_content(self, content: str, section_name: str) -> str:
        """提取章节内容"""
        # 简单的章节提取实现
        lines = content.split('\n')
        section_content = []
        in_section = False

        for line in lines:
            if section_name.lower() in line.lower():
                in_section = True
                continue
            elif in_section and line.strip().startswith('#'):
                # 遇到下一个章节标题，停止
                break
            elif in_section:
                section_content.append(line)

        return '\n'.join(section_content).strip()

    def _build_structured_content(
        self,
        request: ReportGenerationRequest,
        report_data: DocumentReportData | ComparisonReportData | ProjectReportData,
        llm_payload: dict[str, Any]
    ) -> dict[str, Any]:
        full_content = llm_payload.get("full_content", "")
        sections = llm_payload.get("sections", {})
        generated_at = llm_payload.get("generated_at") or datetime.utcnow().isoformat()
        model_used = llm_payload.get("model_used") or request.llm_model

        if request.type == ReportType.DOCUMENT and isinstance(report_data, DocumentReportData):
            structured = self._build_document_structure(report_data)
        elif request.type == ReportType.COMPARISON and isinstance(report_data, ComparisonReportData):
            structured = self._build_comparison_structure(report_data)
        elif request.type == ReportType.PROJECT and isinstance(report_data, ProjectReportData):
            structured = self._build_project_structure(report_data)
        else:
            structured = {}

        return {
            "full_content": full_content,
            "sections": sections,
            "generated_at": generated_at,
            "model_used": model_used,
            "llm_summary": llm_payload,
            "structured_report": structured,
            "metadata": {
                "segment_threshold": self.SIMILARITY_SEGMENT_THRESHOLD,
                "report_type": request.type.value
            }
        }

    def _build_document_structure(self, data: DocumentReportData, include_sources: bool = True) -> dict[str, Any]:
        segments = [
            self._serialize_match_detail(
                match,
                source_label=data.document_title,
                target_label="相似来源"
            )
            for match in data.top_matches
            if match.similarity_score is not None and match.similarity_score >= self.SIMILARITY_SEGMENT_THRESHOLD
        ]

        sources = []
        if include_sources:
            for source in data.sources:
                if source.similarity_score >= self.SIMILARITY_SEGMENT_THRESHOLD:
                    sources.append({
                        "document_id": source.document_id,
                        "document_title": source.document_title,
                        "similarity_score": self._round_score(source.similarity_score),
                        "match_count": source.match_count,
                        "total_text_length": source.total_text_length
                    })

        return {
            "document": {
                "title": data.document_title,
                "total_similarity_score": self._round_score(data.total_similarity_score),
                "risk_level": data.risk_level.value
            },
            "high_similarity_segments": segments,
            "summary": {
                "segment_threshold": self.SIMILARITY_SEGMENT_THRESHOLD,
                "segments_above_threshold": len(segments),
                "statistics": data.statistics,
                "segments_table_html": self._build_similarity_table(segments)
            },
            "top_similarity_sources": sources
        }

    def _build_comparison_structure(self, data: ComparisonReportData) -> dict[str, Any]:
        segments = [
            self._serialize_match_detail(
                match,
                source_label=data.document_a_title,
                target_label=data.document_b_title
            )
            for match in data.match_details
            if match.similarity_score is not None and match.similarity_score >= self.SIMILARITY_SEGMENT_THRESHOLD
        ]

        total_matches = len(data.match_details)
        overflow_warning = None
        if total_matches > self.LARGE_MATCH_DETAIL_LIMIT:
            overflow_warning = (
                f"检测到 {total_matches} 个匹配片段，可能超过LLM处理能力，仅展示阈值以上的重点片段。"
            )

        return {
            "documents": {
                "left": {"id": data.document_a_id, "title": data.document_a_title},
                "right": {"id": data.document_b_id, "title": data.document_b_title}
            },
            "similarity_metrics": {
                "a_to_b": self._round_score(data.similarity_a_to_b),
                "b_to_a": self._round_score(data.similarity_b_to_a),
                "common_similarity": self._round_score(data.common_similarity),
                "unique_a_ratio": self._round_score(data.unique_a_ratio),
                "unique_b_ratio": self._round_score(data.unique_b_ratio)
            },
            "high_similarity_segments": segments,
            "summary": {
                "segment_threshold": self.SIMILARITY_SEGMENT_THRESHOLD,
                "segments_above_threshold": len(segments),
                "segments_table_html": self._build_similarity_table(segments),
                "total_match_details": total_matches,
                "large_dataset_warning": overflow_warning
            },
            "side_by_side_highlights": data.side_by_side_sections
        }

    def _build_project_structure(self, data: ProjectReportData) -> dict[str, Any]:
        high_risk_documents = [
            self._build_document_structure(doc, include_sources=False)
            for doc in data.high_risk_documents
        ]

        statistics_payload = (
            data.statistics.model_dump()
            if hasattr(data.statistics, "model_dump")
            else dict(data.statistics)
        )

        return {
            "project_name": data.project_name,
            "statistics": statistics_payload,
            "high_risk_documents": high_risk_documents,
            "anomalies": data.anomalies,
            "similarity_network": data.similarity_network,
            "recommendations": data.recommendations
        }

    def _serialize_match_detail(
        self,
        match: MatchDetail,
        source_label: str | None = None,
        target_label: str | None = None
    ) -> dict[str, Any]:
        return {
            "similarity_score": self._round_score(match.similarity_score),
            "source": {
                "label": source_label,
                "excerpt": match.source_text,
                "range": {
                    "start": match.source_start,
                    "end": match.source_end
                }
            },
            "target": {
                "label": target_label,
                "excerpt": match.target_text,
                "range": {
                    "start": match.target_start,
                    "end": match.target_end
                }
            },
            "match_type": match.match_type
        }

    def _round_score(self, score: float | None) -> float | None:
        if score is None:
            return None
        return round(float(score), 4)

    def _build_similarity_table(self, segments: list[dict[str, Any]]) -> str:
        if not segments:
            return "<p>未识别到符合阈值的相似片段。</p>"

        header = """
<table class=\"similarity-table\" style=\"width:100%;border-collapse:collapse;margin-top:0.75rem;\">
  <thead>
    <tr style=\"background:#f4f4f5;\">
      <th style=\"border:1px solid #d4d4d8;padding:6px;width:60px;\">序号</th>
      <th style=\"border:1px solid #d4d4d8;padding:6px;width:80px;\">相似度</th>
      <th style=\"border:1px solid #d4d4d8;padding:6px;\">左侧片段</th>
      <th style=\"border:1px solid #d4d4d8;padding:6px;\">右侧片段</th>
    </tr>
  </thead>
  <tbody>
"""

        rows = []
        for index, segment in enumerate(segments, 1):
            similarity = segment.get("similarity_score")
            source = segment.get("source", {})
            target = segment.get("target", {})

            similarity_display = f"{similarity * 100:.2f}%" if similarity is not None else "-"

            left_excerpt = self._format_excerpt(source.get("excerpt"))
            right_excerpt = self._format_excerpt(target.get("excerpt"))

            rows.append(
                "    <tr>\n"
                f"      <td style=\"border:1px solid #d4d4d8;padding:6px;text-align:center;\">{index}</td>\n"
                f"      <td style=\"border:1px solid #d4d4d8;padding:6px;text-align:center;\">{similarity_display}</td>\n"
                f"      <td style=\"border:1px solid #d4d4d8;padding:6px;\">{left_excerpt}</td>\n"
                f"      <td style=\"border:1px solid #d4d4d8;padding:6px;\">{right_excerpt}</td>\n"
                "    </tr>"
            )

        footer = "  </tbody>\n</table>"
        return header + "\n".join(rows) + "\n" + footer

    def _format_excerpt(self, text: str | None) -> str:
        if not text:
            return "<em>无文本</em>"
        safe = html.escape(text)
        return safe.replace("\n", "<br/>")

    def _generate_report_title(self, request: ReportGenerationRequest, data: DocumentReportData | ComparisonReportData | ProjectReportData) -> str:
        """生成报告标题"""
        if request.type == ReportType.DOCUMENT and isinstance(data, DocumentReportData):
            return f"文档抄袭检测报告 - {data.document_title}"
        elif request.type == ReportType.COMPARISON and isinstance(data, ComparisonReportData):
            return f"文档对比分析报告 - {data.document_a_title} vs {data.document_b_title}"
        elif request.type == ReportType.PROJECT and isinstance(data, ProjectReportData):
            return f"项目学术诚信分析报告 - {data.project_name}"
        else:
            return "抄袭检测报告"

    def _extract_summary(self, content: dict[str, Any]) -> str:
        """提取报告摘要"""
        full_content = content.get("full_content", "")
        return full_content.strip() if isinstance(full_content, str) else ""

    def _extract_summary_from_content(self, content: str) -> str:
        """从内容字符串提取摘要"""
        return content.strip()

    def _update_progress(
        self,
        task_id: str,
        progress: float,
        stage: str,
        message: str
    ) -> None:
        """更新进度"""
        if task_id in self.active_generations:
            self.active_generations[task_id].progress = progress
            self.active_generations[task_id].stage = stage
            self.active_generations[task_id].message = message
            self.logger.info(f"Report generation progress: {progress:.1%}", task_id=task_id, stage=stage)

    def get_generation_progress(self, task_id: str) -> ReportProgress | None:
        """获取生成进度"""
        return self.active_generations.get(task_id)

    async def cancel_generation(self, task_id: str) -> bool:
        """取消报告生成"""
        if task_id in self.active_generations:
            del self.active_generations[task_id]
            self.logger.info("Report generation cancelled", task_id=task_id)
            return True
        return False
