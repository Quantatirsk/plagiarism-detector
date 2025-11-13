/**
 * ReportGenerator - 报告生成器组件
 * 提供用户界面来配置和生成各类抄袭检测报告
 */
import { useState, useEffect } from 'react';
import { Card, Button, Form, Select, Switch, InputNumber, Radio, Space, Typography, Alert } from 'antd';
import { FileTextOutlined, EyeOutlined, SettingOutlined, PlayCircleOutlined } from '@ant-design/icons';
import { designSystem } from '@/styles/DesignSystem';

const { Title, Text } = Typography;

// 报告类型配置
export interface ReportConfig {
  type: 'document' | 'comparison' | 'project';
  language: 'zh' | 'en';
  include_charts: boolean;
  include_recommendations: boolean;
  max_matches_detail: number;
  llm_model?: string;
  stream_response: boolean;

  // 类型特定参数
  document_id?: string;
  document_a_id?: string;
  document_b_id?: string;
  project_id?: string;
  include_network_graph?: boolean;
}

interface ReportGeneratorProps {
  onGenerate: (config: ReportConfig) => void;
  onCancel?: () => void;
  isGenerating?: boolean;
  availableDocuments?: Array<{ id: string; title: string }>;
  availableProjects?: Array<{ id: string; name: string }>;
  availableModels?: Array<{ id: string; name: string }>;
  className?: string;
}

export function ReportGenerator({
  onGenerate,
  onCancel,
  isGenerating = false,
  availableDocuments = [],
  availableProjects = [],
  availableModels = [],
}: ReportGeneratorProps) {
  const [form] = Form.useForm();
  const [reportType, setReportType] = useState<'document' | 'comparison' | 'project'>('document');

  // 初始化表单默认值
  useEffect(() => {
    const defaultModel = availableModels[0]?.id || 'google/gemini-2.5-flash-lite';
    const firstDoc = availableDocuments[0]?.id;
    const secondDoc = availableDocuments[1]?.id || firstDoc;
    const firstProject = availableProjects[0]?.id;

    form.setFieldsValue({
      type: 'document',
      language: 'zh',
      include_charts: true,
      include_recommendations: true,
      max_matches_detail: 20,
      stream_response: true,
      include_network_graph: true,
      llm_model: defaultModel,
      document_id: firstDoc,
      document_a_id: firstDoc,
      document_b_id: secondDoc,
      project_id: firstProject,
    });
  }, [availableDocuments, availableProjects, availableModels, form]);

  const handleSubmit = (values: ReportConfig) => {
    onGenerate(values);
  };

  const reportTypes = [
    {
      value: 'document',
      label: '文档抄袭检测报告',
      icon: <FileTextOutlined style={{ fontSize: 20, color: designSystem.colors.primary[500] }} />,
      description: '分析单个文档在数据库中的抄袭情况',
    },
    {
      value: 'comparison',
      label: '文档对比分析报告',
      icon: <EyeOutlined style={{ fontSize: 20, color: designSystem.colors.success[500] }} />,
      description: '深入分析两个文档之间的相似性和差异',
    },
    {
      value: 'project',
      label: '项目学术诚信报告',
      icon: <SettingOutlined style={{ fontSize: 20, color: designSystem.colors.warning[500] }} />,
      description: '对项目内所有文档进行宏观分析',
    },
  ];

  return (
    <div style={{ maxWidth: 800, margin: '0 auto' }}>
      <Card>
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          initialValues={{
            type: 'document',
            language: 'zh',
            include_charts: true,
            include_recommendations: true,
            max_matches_detail: 20,
            stream_response: true,
            include_network_graph: true,
          }}
        >
          {/* 报告类型选择 */}
          <Form.Item
            label={<Text strong>报告类型</Text>}
            name="type"
            rules={[{ required: true, message: '请选择报告类型' }]}
          >
            <Radio.Group
              onChange={(e) => setReportType(e.target.value)}
              style={{ width: '100%' }}
            >
              <Space direction="vertical" style={{ width: '100%' }}>
                {reportTypes.map((type) => (
                  <Radio key={type.value} value={type.value} style={{ width: '100%' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: designSystem.spacing[2] }}>
                      {type.icon}
                      <div>
                        <div style={{ fontWeight: designSystem.typography.fontWeight.medium }}>
                          {type.label}
                        </div>
                        <Text type="secondary" style={{ fontSize: designSystem.typography.fontSize.xs }}>
                          {type.description}
                        </Text>
                      </div>
                    </div>
                  </Radio>
                ))}
              </Space>
            </Radio.Group>
          </Form.Item>

          {/* 文档选择 - 单文档报告 */}
          {reportType === 'document' && (
            <Form.Item
              label="选择文档"
              name="document_id"
              rules={[{ required: true, message: '请选择要分析的文档' }]}
            >
              <Select
                placeholder="请选择文档"
                options={availableDocuments.map(doc => ({
                  label: doc.title,
                  value: doc.id,
                }))}
              />
            </Form.Item>
          )}

          {/* 文档选择 - 对比报告 */}
          {reportType === 'comparison' && (
            <>
              <Form.Item
                label="第一个文档"
                name="document_a_id"
                rules={[{ required: true, message: '请选择第一个对比文档' }]}
              >
                <Select
                  placeholder="请选择文档"
                  options={availableDocuments.map(doc => ({
                    label: doc.title,
                    value: doc.id,
                  }))}
                />
              </Form.Item>
              <Form.Item
                label="第二个文档"
                name="document_b_id"
                rules={[
                  { required: true, message: '请选择第二个对比文档' },
                  ({ getFieldValue }) => ({
                    validator(_, value) {
                      if (!value || getFieldValue('document_a_id') !== value) {
                        return Promise.resolve();
                      }
                      return Promise.reject(new Error('对比文档不能相同'));
                    },
                  }),
                ]}
              >
                <Select
                  placeholder="请选择文档"
                  options={availableDocuments.map(doc => ({
                    label: doc.title,
                    value: doc.id,
                  }))}
                />
              </Form.Item>
            </>
          )}

          {/* 项目选择 - 项目报告 */}
          {reportType === 'project' && (
            <>
              <Form.Item
                label="选择项目"
                name="project_id"
                rules={[{ required: true, message: '请选择要分析的项目' }]}
              >
                <Select
                  placeholder="请选择项目"
                  options={availableProjects.map(proj => ({
                    label: proj.name,
                    value: proj.id,
                  }))}
                />
              </Form.Item>
              <Form.Item
                label="包含网络关系图"
                name="include_network_graph"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>
            </>
          )}

          {/* LLM 模型选择 */}
          <Form.Item
            label="AI 模型"
            name="llm_model"
            tooltip="用于生成报告分析和建议的语言模型"
          >
            <Select
              placeholder="选择 AI 模型"
              options={availableModels.map(model => ({
                label: model.name,
                value: model.id,
              }))}
            />
          </Form.Item>

          {/* 语言选择 */}
          <Form.Item label="报告语言" name="language">
            <Radio.Group>
              <Radio value="zh">中文</Radio>
              <Radio value="en">English</Radio>
            </Radio.Group>
          </Form.Item>

          {/* 高级选项 */}
          <Title level={5} style={{ marginTop: designSystem.spacing[4] }}>高级选项</Title>

          <Form.Item
            label="最大匹配详情数量"
            name="max_matches_detail"
            rules={[
              { required: true, message: '请输入匹配详情数量' },
              { type: 'number', min: 1, max: 100, message: '数量应在 1-100 之间' },
            ]}
          >
            <InputNumber min={1} max={100} style={{ width: '100%' }} />
          </Form.Item>

          <Form.Item
            label="包含图表"
            name="include_charts"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>

          <Form.Item
            label="包含改进建议"
            name="include_recommendations"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>

          <Form.Item
            label="流式响应"
            name="stream_response"
            valuePropName="checked"
            tooltip="启用后可实时查看报告生成进度"
          >
            <Switch />
          </Form.Item>

          {/* 操作按钮 */}
          <Form.Item style={{ marginTop: designSystem.spacing[5], marginBottom: 0 }}>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                icon={<PlayCircleOutlined />}
                loading={isGenerating}
                size="large"
              >
                {isGenerating ? '正在生成...' : '生成报告'}
              </Button>
              {onCancel && (
                <Button onClick={onCancel} disabled={isGenerating} size="large">
                  取消
                </Button>
              )}
            </Space>
          </Form.Item>

          {/* 提示信息 */}
          <Alert
            message="提示"
            description="报告生成可能需要几分钟时间，请耐心等待。生成过程中请不要关闭页面。"
            type="info"
            showIcon
            style={{ marginTop: designSystem.spacing[4] }}
          />
        </Form>
      </Card>
    </div>
  );
}
