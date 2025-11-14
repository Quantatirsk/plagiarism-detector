/**
 * ProjectManagementPage - 项目管理页
 *
 * 功能：
 * - 项目列表展示（卡片网格）
 * - 搜索项目
 * - 创建新项目
 * - 选中项目
 * - 管理项目内的文档
 */

import { useState, useMemo, useEffect } from 'react';
import { Input, Button, Card, Modal, Form, message, Empty, Row, Col, Statistic, Descriptions } from 'antd';
import { PlusOutlined, FolderOutlined, SearchOutlined, FileTextOutlined, CheckCircleOutlined, ArrowLeftOutlined, UnorderedListOutlined } from '@ant-design/icons';
import { useComparisonStore } from '@/store/comparisonStore';
import { useDocuments } from '@/hooks/useData';
import { useComparisonContext } from '@/contexts/useComparisonContext';
import { plagiarismApi } from '@/api/plagiarismApi';
import { LoadingState } from '@/components/common';
import DocumentsTab from '@/pages/workspace/DocumentsTab';
import { designSystem } from '@/styles/DesignSystem';

// 项目统计数据
interface ProjectStats {
  documentCount: number;
  jobCount: number;
}

// 右侧栏组件（使用共享数据）
export function ProjectManagementSidebar(): JSX.Element | null {
  const { selectedProjectId } = useComparisonStore();
  const { projectsState } = useComparisonContext();
  const projects = projectsState.data;
  const [projectStats, setProjectStats] = useState<Record<number, ProjectStats>>({});

  // 加载项目统计数据
  useEffect(() => {
    if (!projects || projects.length === 0) return;

    let mounted = true;
    const loadProjectStats = async () => {
      const stats: Record<number, ProjectStats> = {};
      await Promise.all(
        projects.map(async (project) => {
          try {
            const [docs, jobs] = await Promise.all([
              plagiarismApi.listDocuments({ projectId: project.id }),
              plagiarismApi.listCompareJobs(project.id),
            ]);
            if (mounted) {
              stats[project.id] = {
                documentCount: docs.length,
                jobCount: jobs.length,
              };
            }
          } catch (err) {
            if (mounted) {
              stats[project.id] = { documentCount: 0, jobCount: 0 };
            }
          }
        })
      );
      if (mounted) {
        setProjectStats(stats);
      }
    };
    loadProjectStats();
    return () => { mounted = false; };
  }, [projects]);

  const selectedProject = projects?.find((p) => p.id === selectedProjectId);
  const view = selectedProject ? 'documents' : 'list';

  // 项目列表视图：显示总体统计
  if (view === 'list') {
    const totalProjects = projects?.length ?? 0;
    const totalDocuments = Object.values(projectStats).reduce((sum, stat) => sum + stat.documentCount, 0);
    const totalJobs = Object.values(projectStats).reduce((sum, stat) => sum + stat.jobCount, 0);

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: designSystem.spacing[2] }}>
        <Card size="small" style={{ borderRadius: designSystem.borderRadius.lg }}>
          <Statistic
            title="总项目数"
            value={totalProjects}
            prefix={<FolderOutlined />}
          />
        </Card>
        <Card size="small" style={{ borderRadius: designSystem.borderRadius.lg }}>
          <Statistic
            title="总文档数"
            value={totalDocuments}
            prefix={<FileTextOutlined />}
          />
        </Card>
        <Card size="small" style={{ borderRadius: designSystem.borderRadius.lg }}>
          <Statistic
            title="总任务数"
            value={totalJobs}
            prefix={<UnorderedListOutlined />}
          />
        </Card>
      </div>
    );
  }

  // 文档视图：显示当前项目信息
  if (view === 'documents' && selectedProject) {
    const stats = projectStats[selectedProject.id] ?? { documentCount: 0, jobCount: 0 };

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: designSystem.spacing[2] }}>
        <Card
          title="项目信息"
          size="small"
          style={{ borderRadius: designSystem.borderRadius.lg }}
        >
          <Descriptions column={1} size="small">
            <Descriptions.Item label="项目名称">{selectedProject.name}</Descriptions.Item>
            {selectedProject.description && (
              <Descriptions.Item label="描述">{selectedProject.description}</Descriptions.Item>
            )}
          </Descriptions>
        </Card>
        <Card size="small" style={{ borderRadius: designSystem.borderRadius.lg }}>
          <Statistic
            title="文档数"
            value={stats.documentCount}
            prefix={<FileTextOutlined />}
          />
        </Card>
        <Card size="small" style={{ borderRadius: designSystem.borderRadius.lg }}>
          <Statistic
            title="任务数"
            value={stats.jobCount}
            prefix={<UnorderedListOutlined />}
          />
        </Card>
      </div>
    );
  }

  return null;
}

export default function ProjectManagementPage() {
  // ==================== 状态管理 ====================
  const { selectedProjectId, selectProject } = useComparisonStore();
  const { projectsState } = useComparisonContext();
  const { data: projects, loading, reload } = projectsState;

  // Fix: Memoize filter object to prevent infinite re-renders
  const documentFilter = useMemo(
    () => (selectedProjectId ? { projectId: selectedProjectId } : undefined),
    [selectedProjectId]
  );
  const documentState = useDocuments(documentFilter);

  const [form] = Form.useForm();
  const [searchText, setSearchText] = useState('');
  const [modalOpen, setModalOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [projectStats, setProjectStats] = useState<Record<number, ProjectStats>>({});
  const [view, setView] = useState<'list' | 'documents'>('list');

  // ==================== 加载项目统计数据 ====================
  useEffect(() => {
    if (!projects || projects.length === 0) return;

    let mounted = true;

    const loadProjectStats = async () => {
      const stats: Record<number, ProjectStats> = {};

      await Promise.all(
        projects.map(async (project) => {
          try {
            const [docs, jobs] = await Promise.all([
              plagiarismApi.listDocuments({ projectId: project.id }),
              plagiarismApi.listCompareJobs(project.id),
            ]);

            if (mounted) {
              stats[project.id] = {
                documentCount: docs.length,
                jobCount: jobs.length,
              };
            }
          } catch (err) {
            console.error(`Failed to load stats for project ${project.id}:`, err);
            if (mounted) {
              stats[project.id] = {
                documentCount: 0,
                jobCount: 0,
              };
            }
          }
        })
      );

      if (mounted) {
        setProjectStats(stats);
      }
    };

    loadProjectStats();

    return () => {
      mounted = false;
    };
  }, [projects]);

  // ==================== 筛选逻辑 ====================
  const filteredProjects = useMemo(() => {
    if (!projects) return [];
    if (!searchText) return projects;

    const search = searchText.toLowerCase();
    return projects.filter(
      (p) =>
        (p.name && p.name.toLowerCase().includes(search)) ||
        (p.description && p.description.toLowerCase().includes(search))
    );
  }, [projects, searchText]);

  // ==================== 操作函数 ====================
  const handleCreate = async (values: { name: string; description?: string }) => {
    setCreating(true);
    try {
      const project = await plagiarismApi.createProject({
        name: values.name.trim(),
        description: values.description?.trim() || undefined,
      });

      form.resetFields();
      setModalOpen(false);
      reload();
      message.success('项目创建成功');

      // 自动选中新创建的项目
      selectProject(project.id);
    } catch (err) {
      console.error(err);
      message.error((err as Error).message || '创建项目失败');
    } finally {
      setCreating(false);
    }
  };

  const handleSelectProject = (projectId: number) => {
    selectProject(projectId);
    setView('documents');
  };

  const handleBackToProjects = () => {
    setView('list');
  };

  // ==================== 渲染 ====================
  if (loading) {
    return <LoadingState mode="skeleton" rows={8} />;
  }

  const selectedProject = projects?.find((p) => p.id === selectedProjectId);

  // 文档视图
  if (view === 'documents' && selectedProject) {
    return (
      <div
        style={{
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          gap: designSystem.spacing[1],
        }}
      >
        {/* 顶部导航栏 */}
        <Card
          size="small"
          style={{ borderRadius: designSystem.borderRadius.lg }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: designSystem.spacing[2] }}>
            <Button
              type="text"
              icon={<ArrowLeftOutlined />}
              onClick={handleBackToProjects}
            >
              返回项目列表
            </Button>
            <div style={{ borderLeft: `1px solid ${designSystem.semantic.border.light}`, height: 24 }} />
            <FolderOutlined style={{ fontSize: designSystem.iconSizes.md, color: designSystem.colors.primary[500] }} />
            <span style={{ fontSize: designSystem.typography.fontSize.base, fontWeight: designSystem.typography.fontWeight.semibold }}>
              {selectedProject.name}
            </span>
            {selectedProject.description && (
              <span style={{ fontSize: designSystem.typography.fontSize.sm, color: designSystem.semantic.text.secondary }}>
                - {selectedProject.description}
              </span>
            )}
          </div>
        </Card>

        {/* 文档列表 */}
        <div style={{ flex: 1, minHeight: 0, overflow: 'hidden' }}>
          <DocumentsTab project={selectedProject} documentState={documentState} />
        </div>
      </div>
    );
  }

  // 项目列表视图
  return (
    <div
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        gap: designSystem.spacing[1],
      }}
    >
      {/* 顶部操作区 */}
      <Card
        size="small"
        style={{ borderRadius: designSystem.borderRadius.lg }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: designSystem.spacing[2],
          }}
        >
          <Input
            placeholder="搜索项目..."
            prefix={<SearchOutlined />}
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            allowClear
            style={{ maxWidth: 300 }}
          />
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalOpen(true)}>
            新建项目
          </Button>
        </div>
      </Card>

      {/* 项目列表 */}
      <div
        style={{
          flex: 1,
          minHeight: 0,
          overflow: 'auto',
        }}
      >
        {filteredProjects.length === 0 ? (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description={projects?.length === 0 ? '暂无项目，请创建新项目' : '未找到匹配的项目'}
            style={{ marginTop: designSystem.spacing[8] }}
          >
            {projects?.length === 0 && (
              <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalOpen(true)}>
                创建第一个项目
              </Button>
            )}
          </Empty>
        ) : (
          <Row gutter={[16, 16]}>
            {filteredProjects.map((project) => {
              const isSelected = project.id === selectedProjectId;
              return (
                <Col xs={24} sm={12} lg={8} xl={6} key={project.id}>
                  <Card
                    hoverable
                    onClick={() => handleSelectProject(project.id)}
                    style={{
                      cursor: 'pointer',
                      borderLeft: isSelected
                        ? `3px solid ${designSystem.colors.primary[500]}`
                        : '3px solid transparent',
                      background: isSelected
                        ? designSystem.colors.primary[50]
                        : designSystem.semantic.surface.base,
                      borderRadius: designSystem.borderRadius.lg,
                      height: '100%',
                    }}
                  >
                    <div style={{ display: 'flex', flexDirection: 'column', gap: designSystem.spacing[2] }}>
                      {/* 项目图标和名称 */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: designSystem.spacing[2] }}>
                        <FolderOutlined
                          style={{
                            fontSize: designSystem.iconSizes.xl,
                            color: isSelected
                              ? designSystem.colors.primary[500]
                              : designSystem.colors.primary[300],
                          }}
                        />
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div
                            style={{
                              fontWeight: designSystem.typography.fontWeight.semibold,
                              fontSize: designSystem.typography.fontSize.base,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}
                            title={project.name ?? undefined}
                          >
                            {project.name}
                          </div>
                          <div
                            style={{
                              fontSize: designSystem.typography.fontSize.xs,
                              color: designSystem.semantic.text.tertiary,
                            }}
                          >
                            ID: {project.id}
                          </div>
                        </div>
                      </div>

                      {/* 项目描述 */}
                      {project.description && (
                        <div
                          style={{
                            fontSize: designSystem.typography.fontSize.sm,
                            color: designSystem.semantic.text.secondary,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            display: '-webkit-box',
                            WebkitLineClamp: 2,
                            WebkitBoxOrient: 'vertical',
                            minHeight: '2.5em',
                          }}
                          title={project.description}
                        >
                          {project.description}
                        </div>
                      )}

                      {/* 分隔线 */}
                      <div style={{ borderTop: `1px solid ${designSystem.semantic.border.light}` }} />

                      {/* 项目统计 */}
                      <Row gutter={8}>
                        <Col span={12}>
                          <Statistic
                            title="文档"
                            value={projectStats[project.id]?.documentCount ?? 0}
                            prefix={<FileTextOutlined />}
                            valueStyle={{ fontSize: designSystem.typography.fontSize.base }}
                          />
                        </Col>
                        <Col span={12}>
                          <Statistic
                            title="任务"
                            value={projectStats[project.id]?.jobCount ?? 0}
                            prefix={<CheckCircleOutlined />}
                            valueStyle={{ fontSize: designSystem.typography.fontSize.base }}
                          />
                        </Col>
                      </Row>

                      {/* 更新时间 */}
                      <div
                        style={{
                          fontSize: designSystem.typography.fontSize.xs,
                          color: designSystem.semantic.text.tertiary,
                          textAlign: 'right',
                        }}
                      >
                        更新: {new Date(project.updated_at).toLocaleDateString()}
                      </div>
                    </div>
                  </Card>
                </Col>
              );
            })}
          </Row>
        )}
      </div>

      {/* 创建项目Modal */}
      <Modal
        title="创建新项目"
        open={modalOpen}
        onCancel={() => {
          setModalOpen(false);
          form.resetFields();
        }}
        footer={null}
      >
        <Form form={form} layout="vertical" onFinish={handleCreate}>
          <Form.Item
            label="项目名称"
            name="name"
            rules={[
              { required: true, message: '请输入项目名称' },
              { max: 100, message: '项目名称不能超过100个字符' },
            ]}
          >
            <Input placeholder="请输入项目名称" autoFocus />
          </Form.Item>

          <Form.Item
            label="项目描述"
            name="description"
            rules={[{ max: 500, message: '项目描述不能超过500个字符' }]}
          >
            <Input.TextArea
              placeholder="请输入项目描述（可选）"
              rows={3}
              showCount
              maxLength={500}
            />
          </Form.Item>

          <Form.Item style={{ marginBottom: 0 }}>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: designSystem.spacing[2] }}>
              <Button onClick={() => setModalOpen(false)}>取消</Button>
              <Button type="primary" htmlType="submit" loading={creating}>
                创建
              </Button>
            </div>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}
