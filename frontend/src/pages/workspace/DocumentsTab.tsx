/**
 * DocumentsTab - 文档管理Tab
 * 复用 ProjectDetailPanel 的文档列表和上传功能
 */

import { useState } from 'react';
import { Card, Table, Tag, Upload, message } from 'antd';
import { InboxOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import type { UploadProps } from 'antd';
import { plagiarismApi, type ProjectSummary, type DocumentSummary, type DocumentStatus } from '@/api/plagiarismApi';
import { useDocuments } from '@/hooks/useData';
import { useWorkspaceStore } from '@/store/workspaceStore';
import { DOCUMENT_STATUS_META } from '@/lib/status';
import { useProgressTracking } from '@/hooks/useProgressTracking';
import { ProgressIndicator } from '@/components/progress/ProgressIndicator';
import { designSystem } from '@/styles/DesignSystem';

const { Dragger } = Upload;

interface DocumentsTabProps {
  project: ProjectSummary;
}

export default function DocumentsTab({ project }: DocumentsTabProps) {
  // ==================== 状态管理 ====================
  const [uploading, setUploading] = useState(false);
  const [uploadTaskId, setUploadTaskId] = useState<string | null>(null);
  const { selectedDocumentId, selectDocument } = useWorkspaceStore();

  const documentState = useDocuments({ projectId: project.id });
  const documents = documentState.data ?? [];

  // Progress tracking for uploads
  const uploadProgress = useProgressTracking(uploadTaskId, {
    useSSE: true,
    onComplete: () => {
      setUploadTaskId(null);
      setUploading(false);
      documentState.reload();
    },
    onError: () => {
      setUploadTaskId(null);
      setUploading(false);
      message.error('文件上传失败');
    },
  });

  // ==================== 操作函数 ====================
  const uploadProps: UploadProps = {
    name: 'file',
    multiple: true,
    showUploadList: false,
    beforeUpload: () => false,
    onChange: async (info) => {
      if (info.fileList.length === 0) return;
      setUploading(true);
      try {
        const files = info.fileList.map(f => f.originFileObj).filter(Boolean) as File[];
        const response = await plagiarismApi.uploadDocuments(project.id, files);
        setUploadTaskId(response.task_id);
        documentState.reload();
        message.success('文件上传成功');
      } catch (error) {
        console.error(error);
        message.error((error as Error).message || '上传失败');
        setUploading(false);
      }
    },
  };

  const colorMap = {
    info: 'blue',
    warning: 'orange',
    success: 'green',
    error: 'red',
    neutral: 'default',
  } as const;

  // ==================== 表格列定义 ====================
  const columns: ColumnsType<DocumentSummary> = [
    {
      title: '文档名称',
      dataIndex: 'title',
      key: 'title',
      render: (title: string, record) => (
        <span style={{ fontWeight: designSystem.typography.fontWeight.medium }}>
          {title || record.filename || `文档 #${record.id}`}
        </span>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: DocumentStatus) => {
        const statusMeta = DOCUMENT_STATUS_META[status];
        return <Tag color={colorMap[statusMeta.tone]}>{statusMeta.label}</Tag>;
      },
    },
    {
      title: '语言',
      dataIndex: 'language',
      key: 'language',
      width: 80,
      render: (language: string) => (
        <span style={{ fontSize: designSystem.typography.fontSize.sm, color: designSystem.semantic.text.secondary }}>
          {language || '—'}
        </span>
      ),
    },
    {
      title: '更新时间',
      dataIndex: 'updated_at',
      key: 'updated_at',
      width: 180,
      render: (date: string) => (
        <span style={{ fontSize: designSystem.typography.fontSize.sm, color: designSystem.semantic.text.secondary }}>
          {new Date(date).toLocaleString()}
        </span>
      ),
    },
  ];

  // ==================== 渲染 ====================
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: designSystem.spacing[2],
        height: '100%',
      }}
    >
      {/* 上传区域 */}
      <Card
        size="small"
        title="上传文档"
        style={{ borderRadius: designSystem.borderRadius.lg }}
      >
        <div style={{ marginBottom: designSystem.spacing[2], fontSize: designSystem.typography.fontSize.xs, color: designSystem.semantic.text.secondary }}>
          上传文档到项目，系统会自动解析并处理
        </div>
        <Dragger {...uploadProps} disabled={uploading} style={{ padding: designSystem.spacing[2] }}>
          <p>
            <InboxOutlined style={{ fontSize: 32, color: designSystem.colors.primary[500] }} />
          </p>
          <p style={{ fontSize: designSystem.typography.fontSize.sm }}>
            {uploading ? '正在上传…' : '点击或拖拽选择文件'}
          </p>
          <p style={{ fontSize: designSystem.typography.fontSize.xs, color: designSystem.semantic.text.tertiary }}>
            支持多选，最大 100MB
          </p>
          {uploadProgress.task && (
            <div style={{ marginTop: designSystem.spacing[2] }}>
              <ProgressIndicator task={uploadProgress.task} />
            </div>
          )}
        </Dragger>
      </Card>

      {/* 文档列表 */}
      <Card
        title={`文档列表 (${documents.length})`}
        style={{
          borderRadius: designSystem.borderRadius.lg,
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0,
        }}
        bodyStyle={{
          padding: parseInt(designSystem.spacing[1]),
          flex: 1,
          overflow: 'hidden',
        }}
      >
        <Table
          size="small"
          rowKey="id"
          columns={columns}
          dataSource={documents}
          loading={documentState.loading}
          pagination={{
            pageSize: 20,
            showSizeChanger: true,
            showTotal: (total) => `共 ${total} 条`,
            size: 'small',
          }}
          onRow={(record) => ({
            onClick: () => selectDocument(record.id),
            style: {
              cursor: 'pointer',
              backgroundColor: selectedDocumentId === record.id ? designSystem.colors.primary[50] : undefined
            },
          })}
          scroll={{ y: 'calc(100vh - 520px)' }}
        />
      </Card>
    </div>
  );
}
