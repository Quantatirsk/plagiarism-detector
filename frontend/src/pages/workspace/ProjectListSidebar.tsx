/**
 * ProjectListSidebar - 项目列表侧边栏
 *
 * 功能：
 * - 显示所有项目列表
 * - 支持搜索项目
 * - 创建新项目
 * - 选中项目高亮
 */

import { useState, useMemo } from 'react';
import { Input, Button, Card, Modal, Form, message, Empty } from 'antd';
import { PlusOutlined, FolderOutlined, SearchOutlined } from '@ant-design/icons';
import { useWorkspaceStore } from '@/store/workspaceStore';
import { useProjects } from '@/hooks/useData';
import { plagiarismApi } from '@/api/plagiarismApi';
import { LoadingState } from '@/components/common';
import { designSystem } from '@/styles/DesignSystem';

export default function ProjectListSidebar() {
  // ==================== 状态管理 ====================
  const { selectedProjectId, openProject } = useWorkspaceStore();
  const { data: projects, loading, reload } = useProjects();

  const [form] = Form.useForm();
  const [searchText, setSearchText] = useState('');
  const [modalOpen, setModalOpen] = useState(false);
  const [creating, setCreating] = useState(false);

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
        description: values.description?.trim() || null,
      });

      form.resetFields();
      setModalOpen(false);
      reload();
      message.success('项目创建成功');

      // 自动选中新创建的项目
      openProject(project.id);
    } catch (err) {
      console.error(err);
      message.error((err as Error).message || '创建项目失败');
    } finally {
      setCreating(false);
    }
  };

  // ==================== 渲染 ====================
  if (loading) {
    return <LoadingState mode="skeleton" rows={5} />;
  }

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        gap: designSystem.spacing[1],
      }}
    >
      {/* 搜索框 */}
      <Input
        placeholder="搜索项目..."
        prefix={<SearchOutlined />}
        value={searchText}
        onChange={(e) => setSearchText(e.target.value)}
        allowClear
      />

      {/* 项目列表 */}
      <div style={{ flex: 1, overflow: 'auto' }}>
        {filteredProjects.length === 0 ? (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description={projects?.length === 0 ? '暂无项目' : '未找到匹配的项目'}
            style={{ marginTop: designSystem.spacing[8] }}
          />
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: designSystem.spacing[1] }}>
            {filteredProjects.map((project) => {
              const isSelected = project.id === selectedProjectId;
              return (
                <Card
                  key={project.id}
                  size="small"
                  hoverable
                  onClick={() => openProject(project.id)}
                  style={{
                    cursor: 'pointer',
                    borderLeft: isSelected
                      ? `3px solid ${designSystem.colors.primary[500]}`
                      : '3px solid transparent',
                    background: isSelected
                      ? designSystem.colors.neutral[50]
                      : designSystem.semantic.surface.base,
                    transition: 'all 0.2s',
                  }}
                  bodyStyle={{
                    padding: designSystem.spacing[2],
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: designSystem.spacing[2] }}>
                    <FolderOutlined
                      style={{
                        fontSize: designSystem.iconSizes.md,
                        color: isSelected
                          ? designSystem.colors.primary[500]
                          : designSystem.semantic.text.secondary,
                      }}
                    />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div
                        style={{
                          fontWeight: isSelected
                            ? designSystem.typography.fontWeight.semibold
                            : designSystem.typography.fontWeight.normal,
                          fontSize: designSystem.typography.fontSize.sm,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {project.name}
                      </div>
                      {project.description && (
                        <div
                          style={{
                            fontSize: designSystem.typography.fontSize.xs,
                            color: designSystem.semantic.text.tertiary,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            marginTop: designSystem.spacing[0.5],
                          }}
                        >
                          {project.description}
                        </div>
                      )}
                    </div>
                  </div>
                </Card>
              );
            })}
          </div>
        )}
      </div>

      {/* 创建按钮 */}
      <Button
        type="primary"
        icon={<PlusOutlined />}
        onClick={() => setModalOpen(true)}
        block
      >
        新建项目
      </Button>

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
