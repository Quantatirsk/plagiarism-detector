/**
 * ReportGenerator - 报告生成器组件
 * 提供用户界面来配置和生成各类抄袭检测报告
 */
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  FileText,
  Eye,
  Settings,
  Play,
  AlertCircle,
  Info,
  Loader2
} from 'lucide-react';
import { cn } from '@/lib/utils';

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
  className
}: ReportGeneratorProps) {
  const [config, setConfig] = useState<ReportConfig>({
    type: 'document',
    language: 'zh',
    include_charts: true,
    include_recommendations: true,
    max_matches_detail: 20,
    stream_response: true,
    include_network_graph: true,
    llm_model: 'google/gemini-2.5-flash-lite'
  });

  useEffect(() => {
    setConfig(prev => {
      const next = { ...prev } as ReportConfig;
      let changed = false;

      if (!prev.llm_model && availableModels.length) {
        next.llm_model = availableModels[0]?.id;
        changed = true;
      }
      if (
        availableModels.length &&
        prev.llm_model &&
        !availableModels.some((model) => model.id === prev.llm_model)
      ) {
        next.llm_model = availableModels[0]?.id;
        changed = true;
      }

      if (availableDocuments.length) {
        const firstDoc = availableDocuments[0]?.id;
        const secondDoc = availableDocuments[1]?.id || firstDoc;

        if (!prev.document_id && firstDoc) {
          next.document_id = firstDoc;
          changed = true;
        }

        if (!prev.document_a_id && firstDoc) {
          next.document_a_id = firstDoc;
          changed = true;
        }

        if (!prev.document_b_id && secondDoc) {
          next.document_b_id = secondDoc;
          changed = true;
        }

        if (
          prev.document_b_id &&
          prev.document_a_id &&
          prev.document_a_id === prev.document_b_id &&
          availableDocuments.length > 1
        ) {
          const alternate = availableDocuments.find(doc => doc.id !== prev.document_a_id)?.id;
          if (alternate && alternate !== prev.document_b_id) {
            next.document_b_id = alternate;
            changed = true;
          }
        }
      }

      if (!prev.project_id && availableProjects.length) {
        next.project_id = availableProjects[0]?.id;
        changed = true;
      }

      return changed ? next : prev;
    });
  }, [availableDocuments, availableProjects, availableModels]);

  const [errors, setErrors] = useState<Record<string, string>>({});

  // 验证配置
  const validateConfig = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (config.type === 'document' && !config.document_id) {
      newErrors.document_id = '请选择要分析的文档';
    }

    if (config.type === 'comparison') {
      const hasMultipleDocs = availableDocuments.length > 1;
      if (!config.document_a_id) {
        newErrors.document_a_id = '请选择第一个对比文档';
      }
      if (!config.document_b_id) {
        newErrors.document_b_id = '请选择第二个对比文档';
      }
      if (hasMultipleDocs && config.document_a_id === config.document_b_id) {
        newErrors.document_b_id = '对比文档不能相同';
      }
    }

    if (config.type === 'project' && !config.project_id) {
      newErrors.project_id = '请选择要分析的项目';
    }

    if (config.max_matches_detail < 1 || config.max_matches_detail > 100) {
      newErrors.max_matches_detail = '匹配详情数量应在 1-100 之间';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // 处理生成按钮点击
  const handleGenerate = () => {
    if (validateConfig()) {
      onGenerate(config);
    }
  };

  // 获取报告类型信息
  const getReportTypeInfo = (type: string) => {
    const typeMap = {
      document: {
        title: '文档抄袭检测报告',
        description: '分析单个文档在数据库中的抄袭情况，包括相似度来源、风险评估等',
        icon: FileText,
        color: 'text-blue-600'
      },
      comparison: {
        title: '文档对比分析报告',
        description: '深入分析两个文档之间的相似性和差异，提供并排对比视图',
        icon: Eye,
        color: 'text-green-600'
      },
      project: {
        title: '项目学术诚信报告',
        description: '对项目内所有文档进行宏观分析，识别潜在风险和异常模式',
        icon: Settings,
        color: 'text-purple-600'
      }
    };
    return typeMap[type as keyof typeof typeMap];
  };

  // 渲染报告类型选择
  const renderReportTypeSelector = () => {
    const types = ['document', 'comparison', 'project'] as const;

    return (
      <Card>
        <CardHeader>
          <CardTitle>报告类型</CardTitle>
          <CardDescription>选择要生成的报告类型</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4">
            {types.map((type) => {
              const info = getReportTypeInfo(type);
              const Icon = info.icon;
              const isSelected = config.type === type;

              return (
                <div
                  key={type}
                  className={cn(
                    "p-4 border rounded-lg cursor-pointer transition-colors",
                    isSelected
                      ? "border-primary bg-primary/5"
                      : "border-border hover:border-primary/50"
                  )}
                  onClick={() => setConfig(prev => ({ ...prev, type }))}
                >
                  <div className="flex items-start gap-3">
                    <Icon className={cn("h-5 w-5 mt-1", info.color)} />
                    <div>
                      <h4 className="font-medium mb-1">{info.title}</h4>
                      <p className="text-sm text-muted-foreground">
                        {info.description}
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    );
  };

  // 渲染文档/项目选择
  const renderTargetSelector = () => {
    if (config.type === 'document') {
      return (
        <div className="space-y-2">
          <Label htmlFor="document_id">选择文档</Label>
          <Select
            value={config.document_id || availableDocuments[0]?.id || ''}
            onValueChange={(value) => setConfig(prev => ({ ...prev, document_id: value }))}
          >
            <SelectTrigger className={cn(errors.document_id && "border-red-500")}>
              <SelectValue placeholder="请选择要分析的文档" />
            </SelectTrigger>
            <SelectContent>
              {availableDocuments.map((doc) => (
                <SelectItem key={doc.id} value={doc.id}>
                  {doc.title}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {errors.document_id && (
            <p className="text-sm text-red-500">{errors.document_id}</p>
          )}
        </div>
      );
    }

    if (config.type === 'comparison') {
      return (
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="document_a_id">第一个文档</Label>
            <Select
              value={config.document_a_id || availableDocuments[0]?.id || ''}
              onValueChange={(value) => setConfig(prev => ({ ...prev, document_a_id: value }))}
            >
              <SelectTrigger className={cn(errors.document_a_id && "border-red-500")}>
                <SelectValue placeholder="请选择第一个对比文档" />
              </SelectTrigger>
              <SelectContent>
                {availableDocuments.map((doc) => (
                  <SelectItem key={doc.id} value={doc.id}>
                    {doc.title}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {errors.document_a_id && (
              <p className="text-sm text-red-500">{errors.document_a_id}</p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="document_b_id">第二个文档</Label>
            <Select
              value={config.document_b_id || availableDocuments[1]?.id || availableDocuments[0]?.id || ''}
              onValueChange={(value) => setConfig(prev => ({ ...prev, document_b_id: value }))}
            >
              <SelectTrigger className={cn(errors.document_b_id && "border-red-500")}>
                <SelectValue placeholder="请选择第二个对比文档" />
              </SelectTrigger>
              <SelectContent>
                {(availableDocuments.filter(doc => doc.id !== config.document_a_id).length
                  ? availableDocuments.filter(doc => doc.id !== config.document_a_id)
                  : availableDocuments
                ).map((doc) => (
                  <SelectItem key={doc.id} value={doc.id}>
                    {doc.title}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {errors.document_b_id && (
              <p className="text-sm text-red-500">{errors.document_b_id}</p>
            )}
          </div>
        </div>
      );
    }

    if (config.type === 'project') {
      return (
        <div className="space-y-2">
          <Label htmlFor="project_id">选择项目</Label>
          <Select
            value={config.project_id || availableProjects[0]?.id || ''}
            onValueChange={(value) => setConfig(prev => ({ ...prev, project_id: value }))}
          >
            <SelectTrigger className={cn(errors.project_id && "border-red-500")}>
              <SelectValue placeholder="请选择要分析的项目" />
            </SelectTrigger>
            <SelectContent>
              {availableProjects.map((project) => (
                <SelectItem key={project.id} value={project.id}>
                  {project.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {errors.project_id && (
            <p className="text-sm text-red-500">{errors.project_id}</p>
          )}
        </div>
      );
    }

    return null;
  };

  // 渲染配置选项
  const renderConfigOptions = () => {
    return (
      <Card>
        <CardHeader>
          <CardTitle>生成配置</CardTitle>
          <CardDescription>自定义报告生成选项</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {renderTargetSelector()}

          <div className="space-y-2">
            <Label htmlFor="language">报告语言</Label>
            <Select
              value={config.language}
              onValueChange={(value: 'zh' | 'en') => setConfig(prev => ({ ...prev, language: value }))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="zh">中文</SelectItem>
                <SelectItem value="en">English</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {availableModels.length > 0 && (
            <div className="space-y-2">
              <Label htmlFor="llm_model">LLM 模型（可选）</Label>
              <Select
                value={config.llm_model || ''}
                onValueChange={(value) => setConfig(prev => ({
                  ...prev,
                  llm_model: value || undefined
                }))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="使用默认模型" />
                </SelectTrigger>
                <SelectContent>
                  {availableModels.map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          <div className="space-y-2">
            <Label htmlFor="max_matches_detail">最大匹配详情数量</Label>
            <Input
              id="max_matches_detail"
              type="number"
              min="1"
              max="100"
              value={config.max_matches_detail}
              onChange={(e) => setConfig(prev => ({
                ...prev,
                max_matches_detail: parseInt(e.target.value) || 20
              }))}
              className={cn(errors.max_matches_detail && "border-red-500")}
            />
            {errors.max_matches_detail && (
              <p className="text-sm text-red-500">{errors.max_matches_detail}</p>
            )}
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <Label>包含图表</Label>
                <p className="text-sm text-muted-foreground">
                  在报告中包含数据可视化图表
                </p>
              </div>
              <Switch
                checked={config.include_charts}
                onCheckedChange={(checked) => setConfig(prev => ({
                  ...prev,
                  include_charts: checked
                }))}
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <Label>包含建议</Label>
                <p className="text-sm text-muted-foreground">
                  在报告中包含改进建议和行动计划
                </p>
              </div>
              <Switch
                checked={config.include_recommendations}
                onCheckedChange={(checked) => setConfig(prev => ({
                  ...prev,
                  include_recommendations: checked
                }))}
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <Label>流式响应</Label>
                <p className="text-sm text-muted-foreground">
                  实时显示生成过程，更快看到结果
                </p>
              </div>
              <Switch
                checked={config.stream_response}
                onCheckedChange={(checked) => setConfig(prev => ({
                  ...prev,
                  stream_response: checked
                }))}
              />
            </div>

            {config.type === 'project' && (
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label>包含网络图</Label>
                  <p className="text-sm text-muted-foreground">
                    显示文档间的相似性关系网络
                  </p>
                </div>
                <Switch
                  checked={config.include_network_graph ?? true}
                  onCheckedChange={(checked) => setConfig(prev => ({
                    ...prev,
                    include_network_graph: checked
                  }))}
                />
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    );
  };

  return (
    <div className={cn("space-y-6", className)}>
      {renderReportTypeSelector()}
      {renderConfigOptions()}

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Info className="h-4 w-4" />
          <span>报告生成通常需要 1-3 分钟完成</span>
        </div>

        <div className="flex items-center gap-3">
          {onCancel && (
            <Button variant="outline" onClick={onCancel} disabled={isGenerating}>
              取消
            </Button>
          )}
          <Button
            onClick={handleGenerate}
            disabled={isGenerating}
            className="min-w-[120px]"
          >
            {isGenerating ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                生成中...
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                生成报告
              </>
            )}
          </Button>
        </div>
      </div>

      {Object.keys(errors).length > 0 && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            请修正表单中的错误后重试
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}
