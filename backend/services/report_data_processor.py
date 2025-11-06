"""
报告数据处理服务 - 处理和聚合抄袭检测数据为报告生成做准备
"""
from typing import List, Dict, Any, Optional
from collections import defaultdict
import statistics

from backend.models.report_models import (
    DocumentReportData, ComparisonReportData, ProjectReportData,
    SimilaritySource, MatchDetail, ProjectStatistics, RiskLevel
)
from backend.api.v1.compare import PairReportResponse, MatchGroup, MatchDetailModel, PairResponse
from backend.db.models import Document, Project, ComparePair, CompareJob, DocumentChunk
from backend.services.base_service import BaseService, singleton
from backend.db import get_session
from sqlmodel import select, col
from sqlmodel.ext.asyncio.session import AsyncSession
from backend.services.service_factory import ServiceFactory


@singleton
class ReportDataProcessor(BaseService):
    """报告数据处理服务"""

    def _initialize(self):
        """初始化数据处理器"""
        self.risk_thresholds = {
            RiskLevel.LOW: (0.0, 0.2),
            RiskLevel.MODERATE: (0.2, 0.5),
            RiskLevel.HIGH: (0.5, 0.8),
            RiskLevel.CRITICAL: (0.8, 1.0)
        }
        self.orchestrator = ServiceFactory.get_detection_orchestrator()

    def calculate_risk_level(self, similarity_score: float) -> RiskLevel:
        """根据相似度计算风险等级"""
        for level, (min_score, max_score) in self.risk_thresholds.items():
            if min_score <= similarity_score < max_score:
                return level
        return RiskLevel.CRITICAL  # 默认为最高风险

    async def process_document_report_data(
        self,
        document_id: str | int,
        max_matches: int = 20
    ) -> DocumentReportData:
        """处理单文档报告数据"""
        self._ensure_initialized()
        async with get_session() as session:
            # 获取文档信息
            document = await self._get_document(session, document_id)
            if not document:
                raise ValueError(f"Document not found: {document_id}")

            # 获取所有相关的比较结果
            pair_reports = await self._get_document_comparisons(session, document_id)

            # 聚合相似度数据
            sources = self._aggregate_similarity_sources(pair_reports, document_id)

            # 计算总体相似度
            total_similarity = self._calculate_total_similarity(sources)

            # 提取高匹配详情
            top_matches = self._extract_top_matches(pair_reports, max_matches)

            # 计算统计信息
            statistics = self._calculate_document_statistics(pair_reports, sources)

            return DocumentReportData(
                document_id=str(document_id),
                document_title=document.title or f"Document {document_id}",
                total_similarity_score=total_similarity,
                risk_level=self.calculate_risk_level(total_similarity),
                sources=sources,
                top_matches=top_matches,
                statistics=statistics
            )

    async def process_comparison_report_data(
        self,
        document_a_id: str,
        document_b_id: str
    ) -> ComparisonReportData:
        """处理双文档对比报告数据"""
        self._ensure_initialized()
        async with get_session() as session:
            # 获取文档信息
            doc_a = await self._get_document(session, document_a_id)
            doc_b = await self._get_document(session, document_b_id)

            if not doc_a or not doc_b:
                raise ValueError("One or both documents not found")

            # 获取比较结果
            pair_report_ab = await self._get_pair_report(session, document_a_id, document_b_id)

            if not pair_report_ab:
                raise ValueError(f"Comparison not found between {document_a_id} and {document_b_id}")

            # 计算相似度指标
            similarity = self._calculate_directional_similarity(pair_report_ab)
            similarity_a_to_b = similarity
            similarity_b_to_a = similarity

            # 计算内容分布
            common_similarity = similarity
            unique_a_ratio = max(0.0, 1.0 - similarity)
            unique_b_ratio = max(0.0, 1.0 - similarity)

            # 提取匹配详情
            match_details = [self._convert_detail_model(detail) for detail in pair_report_ab.details]

            # 生成并排对比数据
            side_by_side_sections = self._generate_side_by_side_data(pair_report_ab)

            return ComparisonReportData(
                document_a_id=document_a_id,
                document_b_id=document_b_id,
                document_a_title=doc_a.title or f"Document {document_a_id}",
                document_b_title=doc_b.title or f"Document {document_b_id}",
                similarity_a_to_b=similarity_a_to_b,
                similarity_b_to_a=similarity_b_to_a,
                common_similarity=common_similarity,
                unique_a_ratio=unique_a_ratio,
                unique_b_ratio=unique_b_ratio,
                match_details=match_details,
                side_by_side_sections=side_by_side_sections
            )

    async def process_project_report_data(
        self,
        project_id: str,
        include_network: bool = True
    ) -> ProjectReportData:
        """处理项目级报告数据"""
        self._ensure_initialized()
        async with get_session() as session:
            # 获取项目信息
            project = await self._get_project(session, project_id)
            if not project:
                raise ValueError(f"Project not found: {project_id}")

            # 获取项目内所有文档
            documents = await self._get_project_documents(session, project_id)

            # 获取所有文档间的比较结果
            all_comparisons = await self._get_project_comparisons(session, project_id)

            # 计算项目统计信息
            statistics = self._calculate_project_statistics(documents, all_comparisons)

            # 识别高风险文档
            high_risk_documents = await self._identify_high_risk_documents(
                documents, all_comparisons
            )

            # 生成相似度网络图数据
            similarity_network = self._generate_similarity_network(documents, all_comparisons) if include_network else {}

            # 异常检测
            anomalies = self._detect_anomalies(all_comparisons)

            # 生成建议
            recommendations = self._generate_project_recommendations(statistics, anomalies)

            return ProjectReportData(
                project_id=project_id,
                project_name=project.name or f"Project {project_id}",
                statistics=statistics,
                high_risk_documents=high_risk_documents,
                similarity_network=similarity_network,
                anomalies=anomalies,
                recommendations=recommendations
            )

    def _aggregate_similarity_sources(
        self,
        pair_reports: List[PairReportResponse],
        target_document_id: str | int
    ) -> List[SimilaritySource]:
        """聚合相似度来源"""
        target_id = self._parse_int_id(target_document_id, "document")
        source_map: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
            'similarity_scores': [],
            'match_count': 0,
            'total_length': 0
        })

        for report in pair_reports:
            # 根据目标文档确定来源文档ID
            if report.left_document_id == target_id:
                source_id = report.right_document_id
            elif report.right_document_id == target_id:
                source_id = report.left_document_id
            else:
                # 如果报告与目标文档无关则跳过
                continue

            # 计算相似度
            similarity = self._calculate_directional_similarity(report)
            source_map[source_id]['similarity_scores'].append(similarity)
            source_map[source_id]['match_count'] += len(report.details)

            # 计算覆盖长度
            for detail in report.details:
                source_map[source_id]['total_length'] += self._estimate_span_coverage(detail)

        # 转换为SimilaritySource对象
        sources = []
        for source_id, data in source_map.items():
            if data['similarity_scores']:
                avg_similarity = statistics.mean(data['similarity_scores'])
                sources.append(SimilaritySource(
                    document_id=str(source_id),
                    document_title=f"Document {source_id}",
                    similarity_score=avg_similarity,
                    match_count=data['match_count'],
                    total_text_length=data['total_length']
                ))

        # 按相似度排序
        sources.sort(key=lambda x: x.similarity_score, reverse=True)
        return sources

    def _calculate_total_similarity(self, sources: List[SimilaritySource]) -> float:
        """计算总体相似度"""
        if not sources:
            return 0.0

        # 使用加权平均，权重基于匹配文本长度
        total_weight = sum(source.total_text_length for source in sources)
        if total_weight == 0:
            return statistics.mean(source.similarity_score for source in sources)

        weighted_sum = sum(
            source.similarity_score * source.total_text_length
            for source in sources
        )
        return weighted_sum / total_weight

    def _extract_top_matches(
        self,
        pair_reports: List[PairReportResponse],
        max_matches: int
    ) -> List[MatchDetail]:
        """提取高匹配详情"""
        all_matches = []

        for report in pair_reports:
            for detail in report.details:
                all_matches.append(self._convert_detail_model(detail))

        # 按相似度排序并取前N个
        all_matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return all_matches[:max_matches]

    def _calculate_document_statistics(
        self,
        pair_reports: List[PairReportResponse],
        sources: List[SimilaritySource]
    ) -> Dict[str, Any]:
        """计算文档统计信息"""
        similarities = [source.similarity_score for source in sources]

        return {
            'total_sources': len(sources),
            'total_comparisons': len(pair_reports),
            'average_similarity': statistics.mean(similarities) if similarities else 0,
            'max_similarity': max(similarities) if similarities else 0,
            'min_similarity': min(similarities) if similarities else 0,
            'similarity_std': statistics.stdev(similarities) if len(similarities) > 1 else 0,
            'high_risk_sources': sum(1 for s in similarities if s > 0.7),
            'similarity_distribution': self._create_distribution(similarities)
        }

    def _calculate_project_statistics(
        self,
        documents: List[Document],
        comparisons: List[PairReportResponse]
    ) -> ProjectStatistics:
        """计算项目统计信息"""
        if not comparisons:
            return ProjectStatistics(
                total_documents=len(documents),
                total_comparisons=0,
                average_similarity=0,
                high_risk_count=0,
                similarity_distribution={},
                most_similar_pairs=[]
            )

        similarities = [self._calculate_directional_similarity(comp) for comp in comparisons]
        high_risk_count = sum(1 for s in similarities if s > 0.7)

        # 找出最相似的文档对
        similarity_pairs = [
            {
                'document_a': comp.left_document_id,
                'document_b': comp.right_document_id,
                'similarity': self._calculate_directional_similarity(comp)
            }
            for comp in comparisons
        ]
        most_similar_pairs = sorted(
            similarity_pairs,
            key=lambda x: x['similarity'],
            reverse=True
        )[:10]

        return ProjectStatistics(
            total_documents=len(documents),
            total_comparisons=len(comparisons),
            average_similarity=statistics.mean(similarities),
            high_risk_count=high_risk_count,
            similarity_distribution=self._create_distribution(similarities),
            most_similar_pairs=most_similar_pairs
        )

    def _create_distribution(self, values: List[float], bins: int = 10) -> Dict[str, int]:
        """创建数值分布"""
        if not values:
            return {}

        min_val, max_val = 0, 1  # 相似度范围是0-1
        bin_width = (max_val - min_val) / bins

        distribution = {}
        for i in range(bins):
            bin_start = min_val + i * bin_width
            bin_end = bin_start + bin_width
            bin_label = f"{bin_start:.1f}-{bin_end:.1f}"

            count = sum(1 for v in values if bin_start <= v < bin_end)
            # 最后一个bin包含最大值
            if i == bins - 1:
                count = sum(1 for v in values if bin_start <= v <= bin_end)

            distribution[bin_label] = count

        return distribution

    def _calculate_directional_similarity(self, pair_report: PairReportResponse) -> float:
        """计算方向性相似度"""
        if not pair_report.details:
            return 0.0

        total_similarity = sum(self._detail_similarity(detail) for detail in pair_report.details)
        total_matches = len(pair_report.details)
        return total_similarity / total_matches if total_matches > 0 else 0.0

    # 数据库访问辅助方法
    def _parse_int_id(self, value: str | int, entity: str) -> int:
        """Convert incoming IDs to integers, raising a helpful error on failure."""
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {entity} id: {value}") from exc

    async def _get_document(self, session: AsyncSession, document_id: str | int) -> Optional[Document]:
        """获取文档"""
        doc_id = self._parse_int_id(document_id, "document")
        result = await session.exec(select(Document).where(Document.id == doc_id))
        return result.first()

    async def _get_project(self, session: AsyncSession, project_id: str | int) -> Optional[Project]:
        """获取项目"""
        proj_id = self._parse_int_id(project_id, "project")
        result = await session.exec(select(Project).where(Project.id == proj_id))
        return result.first()

    async def _get_document_comparisons(self, session: AsyncSession, document_id: str | int) -> List[PairReportResponse]:
        """获取文档的所有比较结果"""
        doc_id = self._parse_int_id(document_id, "document")

        stmt = (
            select(ComparePair)
            .where((col(ComparePair.left_document_id) == doc_id) | (col(ComparePair.right_document_id) == doc_id))
            .order_by(col(ComparePair.created_at).desc())
        )
        result = await session.exec(stmt)
        pairs = result.all()

        reports: List[PairReportResponse] = []
        for pair in pairs:
            try:
                report = await self._build_pair_report_response(session, pair)
                reports.append(report)
            except ValueError as e:
                self.logger.warning(f"Skipping invalid pair {pair.id}: {e}")
                continue

        return reports

    async def _build_pair_report_response(self, session: AsyncSession, pair: ComparePair) -> PairReportResponse:
        self._ensure_initialized()
        if pair.id is None:
            raise ValueError("Pair has no identifier")

        report = await self.orchestrator.fetch_compare_report(pair.id)
        if not report.left_document or not report.right_document:
            raise ValueError("Comparison report missing document references")

        # Fetch chunk texts for detailed excerpts
        chunk_ids: set[int] = {
            detail.left_chunk_id for detail in report.details if detail.left_chunk_id is not None
        } | {
            detail.right_chunk_id for detail in report.details if detail.right_chunk_id is not None
        }

        chunk_text_map: Dict[int, str] = {}
        if chunk_ids:
            result = await session.exec(select(DocumentChunk).where(col(DocumentChunk.id).in_(list(chunk_ids))))
            chunk_text_map = {chunk.id: chunk.text for chunk in result.all() if chunk.id is not None}

        if report.pair.id is None:
            raise ValueError("Pair ID is required")
        if report.pair.left_document_id is None or report.pair.right_document_id is None:
            raise ValueError("Document IDs are required")

        pair_payload = PairResponse(
            id=report.pair.id,
            job_id=report.pair.job_id,
            left_document_id=report.pair.left_document_id,
            right_document_id=report.pair.right_document_id,
            status=report.pair.status,
            metrics=report.pair.metrics_json,
        )

        groups = [
            MatchGroup(
                id=group.id or 0,
                left_chunk_id=group.left_chunk_id,
                right_chunk_id=group.right_chunk_id,
                final_score=group.final_score,
                semantic_score=group.semantic_score,
                cross_score=group.cross_score,
                alignment_ratio=group.alignment_ratio,
                span_count=group.span_count,
                match_count=group.match_count,
                # FIXME: Database model returns dict instead of list, wrap it temporarily
                paragraph_spans=[group.paragraph_spans_json] if isinstance(group.paragraph_spans_json, dict) else group.paragraph_spans_json,
                document_spans=[group.document_spans_json] if isinstance(group.document_spans_json, dict) else group.document_spans_json,
            )
            for group in report.groups
        ]

        def build_excerpt(chunk_id: Optional[int]) -> Optional[str]:
            if chunk_id is None:
                return None
            text = chunk_text_map.get(chunk_id)
            return self._excerpt_text(text)

        details = []
        for detail in report.details:
            details.append(
                MatchDetailModel(
                    group_id=detail.group_id,
                    left_chunk_id=detail.left_chunk_id,
                    right_chunk_id=detail.right_chunk_id,
                    final_score=detail.final_score,
                    semantic_score=detail.semantic_score,
                    cross_score=detail.cross_score,
                    # FIXME: Database model returns dict instead of list, wrap it temporarily
                    spans=[detail.spans_json] if isinstance(detail.spans_json, dict) else detail.spans_json,
                    left_excerpt=build_excerpt(detail.left_chunk_id),
                    right_excerpt=build_excerpt(detail.right_chunk_id)
                )
            )

        if report.left_document.id is None or report.right_document.id is None:
            raise ValueError("Document IDs are required")

        return PairReportResponse(
            pair=pair_payload,
            left_document_id=report.left_document.id,
            right_document_id=report.right_document.id,
            groups=groups,
            details=details,
        )

    async def _get_pair_report(
        self,
        session: AsyncSession,
        doc_a_id: str | int,
        doc_b_id: str | int
    ) -> Optional[PairReportResponse]:
        """获取特定文档对的比较结果"""
        left_id = self._parse_int_id(doc_a_id, "document")
        right_id = self._parse_int_id(doc_b_id, "document")

        stmt = (
            select(ComparePair)
            .where(col(ComparePair.left_document_id) == left_id)
            .where(col(ComparePair.right_document_id) == right_id)
            .order_by(col(ComparePair.created_at).desc())
        )
        result = await session.exec(stmt)
        pair = result.first()

        if not pair:
            return None

        try:
            return await self._build_pair_report_response(session, pair)
        except ValueError as exc:
            self.logger.warning(
                "Failed to build pair report",
                pair_id=pair.id,
                error=str(exc)
            )
            return None

    async def _get_project_documents(self, session: AsyncSession, project_id: str | int) -> List[Document]:
        """获取项目内所有文档"""
        proj_id = self._parse_int_id(project_id, "project")
        result = await session.exec(select(Document).where(Document.project_id == proj_id))
        return list(result.all())

    async def _get_project_comparisons(self, session: AsyncSession, project_id: str | int) -> List[PairReportResponse]:
        """获取项目内所有比较结果"""
        proj_id = self._parse_int_id(project_id, "project")

        stmt = (
            select(ComparePair)
            .join(CompareJob, col(ComparePair.job_id) == col(CompareJob.id))
            .where(col(CompareJob.project_id) == proj_id)
            .order_by(col(ComparePair.created_at).desc())
        )
        result = await session.exec(stmt)
        pairs = result.all()

        reports: List[PairReportResponse] = []
        for pair in pairs:
            try:
                report = await self._build_pair_report_response(session, pair)
                reports.append(report)
            except ValueError as e:
                self.logger.warning(f"Skipping invalid pair {pair.id}: {e}")
                continue

        return reports

    def _excerpt_text(self, text: Optional[str], max_length: Optional[int] = None) -> Optional[str]:
        """提取文本摘录，默认不截断以显示完整段落"""
        if not text:
            return None
        clean = text.strip()
        if max_length is None or len(clean) <= max_length:
            return clean
        return clean[:max_length].rstrip() + "..."

    def _group_details_by_group(self, report: PairReportResponse) -> Dict[int, List[MatchDetailModel]]:
        """按匹配组聚合详情数据"""
        detail_map: Dict[int, List[MatchDetailModel]] = defaultdict(list)
        for detail in report.details:
            detail_map[detail.group_id].append(detail)
        return detail_map

    def _detail_similarity(self, detail: Optional[MatchDetailModel]) -> float:
        """提取匹配详情的相似度分数"""
        if detail is None:
            return 0.0
        for value in (detail.final_score, detail.semantic_score, detail.cross_score):
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    def _estimate_span_coverage(self, detail: MatchDetailModel) -> int:
        """根据span估算匹配覆盖长度"""
        coverage = 0
        spans = detail.spans or []
        if isinstance(spans, list):
            for span in spans:
                if not isinstance(span, dict):
                    continue
                left_start = span.get('left_start', 0)
                left_end = span.get('left_end', left_start)
                right_start = span.get('right_start', 0)
                right_end = span.get('right_end', right_start)
                left_len = abs(int(left_end) - int(left_start)) if left_end is not None and left_start is not None else 0
                right_len = abs(int(right_end) - int(right_start)) if right_end is not None and right_start is not None else 0
                coverage += max(left_len, right_len)
        return coverage

    def _convert_detail_model(self, detail: MatchDetailModel) -> MatchDetail:
        """将MatchDetailModel转换为报告使用的MatchDetail结构"""
        spans = detail.spans or []
        left_start = left_end = right_start = right_end = 0
        if isinstance(spans, list) and spans:
            first = spans[0]
            if isinstance(first, dict):
                left_start = int(first.get('left_start', 0) or 0)
                left_end = int(first.get('left_end', left_start) or left_start)
                right_start = int(first.get('right_start', 0) or 0)
                right_end = int(first.get('right_end', right_start) or right_start)

        return MatchDetail(
            source_text=(detail.left_excerpt or f"Chunk {detail.left_chunk_id}"),
            target_text=(detail.right_excerpt or f"Chunk {detail.right_chunk_id}"),
            similarity_score=self._detail_similarity(detail),
            source_start=left_start,
            source_end=left_end,
            target_start=right_start,
            target_end=right_end,
            match_type='semantic'
        )

    def _generate_side_by_side_data(
        self,
        pair_report: PairReportResponse
    ) -> List[Dict[str, Any]]:
        """生成并排对比数据"""
        sections = []
        detail_map = self._group_details_by_group(pair_report)

        for index, group in enumerate(pair_report.groups):
            group_details = detail_map.get(group.id, [])
            detail = group_details[0] if group_details else None

            similarity = self._detail_similarity(detail) if detail else (group.final_score or 0.0)

            left_start = left_end = right_start = right_end = 0
            if detail and isinstance(detail.spans, list) and detail.spans:
                first_span = detail.spans[0]
                if isinstance(first_span, dict):
                    left_start = int(first_span.get('left_start', 0) or 0)
                    left_end = int(first_span.get('left_end', left_start) or left_start)
                    right_start = int(first_span.get('right_start', 0) or 0)
                    right_end = int(first_span.get('right_end', right_start) or right_start)

            sections.append({
                'section_id': group.id or index,
                'text_a': f"关联段落 {detail.left_chunk_id if detail and detail.left_chunk_id is not None else group.left_chunk_id}",
                'text_b': f"关联段落 {detail.right_chunk_id if detail and detail.right_chunk_id is not None else group.right_chunk_id}",
                'similarity': similarity,
                'positions_a': (left_start, left_end),
                'positions_b': (right_start, right_end)
            })
        return sections

    async def _identify_high_risk_documents(
        self,
        documents: List[Document],
        comparisons: List[PairReportResponse]
    ) -> List[DocumentReportData]:
        """识别高风险文档"""
        high_risk_docs = []

        # 计算每个文档的风险
        for doc in documents:
            if doc.id is None:
                continue

            doc_comparisons = [
                comp for comp in comparisons
                if comp.left_document_id == doc.id or comp.right_document_id == doc.id
            ]

            if doc_comparisons:
                doc_report_data = await self.process_document_report_data(doc.id)
                if doc_report_data.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    high_risk_docs.append(doc_report_data)

        return sorted(high_risk_docs, key=lambda x: x.total_similarity_score, reverse=True)

    def _generate_similarity_network(
        self,
        documents: List[Document],
        comparisons: List[PairReportResponse]
    ) -> Dict[str, Any]:
        """生成相似度网络图数据"""
        nodes = [
            {'id': doc.id, 'label': doc.title, 'size': 1}
            for doc in documents
        ]

        edges = []
        for comp in comparisons:
            similarity = self._calculate_directional_similarity(comp)
            if similarity > 0.3:  # 只显示相似度较高的连接
                edges.append({
                    'source': comp.left_document_id,
                    'target': comp.right_document_id,
                    'weight': similarity,
                    'label': f'{similarity:.1%}'
                })

        return {
            'nodes': nodes,
            'edges': edges,
            'layout': 'force',
            'config': {
                'node_size_field': 'size',
                'edge_width_field': 'weight',
                'show_labels': True
            }
        }

    def _detect_anomalies(
        self,
        comparisons: List[PairReportResponse]
    ) -> List[Dict[str, Any]]:
        """检测异常高相似度"""
        anomalies = []

        if not comparisons:
            return anomalies

        # 检测高相似度（>0.8）
        for comp in comparisons:
            similarity = self._calculate_directional_similarity(comp)
            if similarity > 0.8:
                anomalies.append({
                    'type': 'high_similarity',
                    'severity': 'critical' if similarity > 0.9 else 'high',
                    'description': f'异常高相似度: {similarity:.1%}',
                    'documents': [comp.left_document_id, comp.right_document_id],
                    'similarity': similarity
                })

        return anomalies

    def _generate_project_recommendations(
        self,
        statistics: ProjectStatistics,
        anomalies: List[Dict[str, Any]]
    ) -> List[str]:
        """生成项目建议"""
        recommendations = []

        # 基于统计数据的建议
        if statistics.average_similarity > 0.3:
            recommendations.append("项目整体相似度偏高，建议加强原创性教育")

        if statistics.high_risk_count > statistics.total_documents * 0.1:
            recommendations.append("高风险文档比例较高，建议进行个别辅导")

        # 基于异常的建议
        if anomalies:
            critical_anomalies = [a for a in anomalies if a['severity'] == 'critical']
            if critical_anomalies:
                recommendations.append("发现严重抄袭嫌疑，建议立即进行人工审核")

        if len(recommendations) == 0:
            recommendations.append("项目整体学术诚信状况良好，建议继续保持")

        return recommendations
