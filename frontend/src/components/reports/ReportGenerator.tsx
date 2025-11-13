/**
 * ReportGenerator - 报告生成器组件
 * 提供用户界面来配置和生成各类抄袭检测报告
 */
import { useState, useEffect } from 'react';
import { Button, Form, Select, Switch, InputNumber, Segmented } from 'antd';
import { PlayCircleOutlined } from '@ant-design/icons';
import { designSystem } from '@/styles/DesignSystem';

// 报告类型配置
export interface ReportConfig {
  type: 'document' | 'comparison';
  include_charts: boolean;
  include_recommendations: boolean;
  max_matches_detail: number;
  llm_model?: string;
  stream_response: boolean;

  // 类型特定参数
  document_id?: string;
  document_a_id?: string;
  document_b_id?: string;
}

interface ReportGeneratorProps {
  onGenerate: (config: ReportConfig) => void;
  isGenerating?: boolean;
  availableDocuments?: Array<{ id: string; title: string }>;
  availableModels?: Array<{ id: string; name: string }>;
  className?: string;
}

export function ReportGenerator({
  onGenerate,
  isGenerating = false,
  availableDocuments = [],
  availableModels = [],
}: ReportGeneratorProps) {
  const [form] = Form.useForm();
  const [reportType, setReportType] = useState<'document' | 'comparison'>('comparison');

  // 初始化表单默认值
  useEffect(() => {
    const defaultModel = availableModels[0]?.id || 'google/gemini-2.5-flash-lite';
    const firstDoc = availableDocuments[0]?.id;
    const secondDoc = availableDocuments[1]?.id || firstDoc;

    form.setFieldsValue({
      type: 'comparison',
      include_charts: true,
      include_recommendations: true,
      max_matches_detail: 20,
      stream_response: true,
      llm_model: defaultModel,
      document_id: firstDoc,
      document_a_id: firstDoc,
      document_b_id: secondDoc,
    });
  }, [availableDocuments, availableModels, form]);

  const handleSubmit = (values: ReportConfig) => {
    onGenerate(values);
  };

  return (
    <Form
      form={form}
      layout="inline"
      onFinish={handleSubmit}
      initialValues={{
        type: 'comparison',
        include_charts: true,
        include_recommendations: true,
        max_matches_detail: 20,
        stream_response: true,
      }}
      style={{ marginBottom: 0, display: 'flex', flexWrap: 'wrap', gap: designSystem.spacing[2], alignItems: 'center' }}
    >
      {/* 报告类型 */}
      <Form.Item
        name="type"
        rules={[{ required: true }]}
        style={{ marginBottom: 0 }}
      >
        <Segmented
          options={[
            { label: '文档对比', value: 'comparison' },
            { label: '文档检测', value: 'document' },
          ]}
          onChange={(value) => setReportType(value as 'document' | 'comparison')}
          size="small"
        />
      </Form.Item>

      {/* 文档选择 - 单文档 */}
      {reportType === 'document' && (
        <Form.Item
          name="document_id"
          rules={[{ required: true }]}
          style={{ marginBottom: 0, width: 200 }}
        >
          <Select
            placeholder="选择文档"
            size="small"
            options={availableDocuments.map(doc => ({
              label: doc.title,
              value: doc.id,
            }))}
          />
        </Form.Item>
      )}

      {/* 文档选择 - 对比 */}
      {reportType === 'comparison' && (
        <>
          <Form.Item
            name="document_a_id"
            rules={[{ required: true }]}
            style={{ marginBottom: 0, width: 160 }}
          >
            <Select
              placeholder="文档 A"
              size="small"
              options={availableDocuments.map(doc => ({
                label: doc.title,
                value: doc.id,
              }))}
            />
          </Form.Item>
          <Form.Item
            name="document_b_id"
            rules={[
              { required: true },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  if (!value || getFieldValue('document_a_id') !== value) {
                    return Promise.resolve();
                  }
                  return Promise.reject(new Error('不能相同'));
                },
              }),
            ]}
            style={{ marginBottom: 0, width: 160 }}
          >
            <Select
              placeholder="文档 B"
              size="small"
              options={availableDocuments.map(doc => ({
                label: doc.title,
                value: doc.id,
              }))}
            />
          </Form.Item>
        </>
      )}

      <Form.Item
        name="llm_model"
        style={{ marginBottom: 0, width: 160 }}
      >
        <Select
          placeholder="AI 模型"
          size="small"
          options={availableModels.map(model => ({
            label: model.name,
            value: model.id,
          }))}
        />
      </Form.Item>

      {/* 高级选项 */}
      <Form.Item
        name="max_matches_detail"
        style={{ marginBottom: 0, width: 80 }}
      >
        <InputNumber
          min={1}
          max={100}
          size="small"
          placeholder="匹配"
          style={{ width: '100%' }}
        />
      </Form.Item>

      <Form.Item name="include_charts" valuePropName="checked" style={{ marginBottom: 0 }}>
        <Switch size="small" checkedChildren="图" unCheckedChildren="图" />
      </Form.Item>

      <Form.Item name="include_recommendations" valuePropName="checked" style={{ marginBottom: 0 }}>
        <Switch size="small" checkedChildren="建议" unCheckedChildren="建议" />
      </Form.Item>

      <Form.Item name="stream_response" valuePropName="checked" style={{ marginBottom: 0 }}>
        <Switch size="small" checkedChildren="流式" unCheckedChildren="流式" />
      </Form.Item>

      <Form.Item style={{ marginBottom: 0, marginLeft: 'auto' }}>
        <Button
          type="primary"
          htmlType="submit"
          icon={<PlayCircleOutlined />}
          loading={isGenerating}
          size="small"
        >
          {isGenerating ? '生成中' : '生成'}
        </Button>
      </Form.Item>
    </Form>
  );
}
