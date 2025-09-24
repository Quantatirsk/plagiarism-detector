import { useState } from 'react';
import type { ChangeEvent, FormEvent } from 'react';
import { plagiarismApi, type ProjectSummary } from '@/api/plagiarismApi';
import { useProjects } from '@/hooks/useData';
import { PageShell, PageHeader, PageContent, SectionCard } from '@/components/layout/Page';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { TABLE_BASE, TABLE_BODY, TABLE_HEAD } from '@/lib/table';

interface ProjectsPageProps {
  onSelectProject: (project: ProjectSummary) => void;
}

export function ProjectsPage({ onSelectProject }: ProjectsPageProps) {
  const { data: projects, loading, error, reload } = useProjects();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [creating, setCreating] = useState(false);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!name.trim()) {
      alert('请填写项目名称');
      return;
    }
    setCreating(true);
    try {
      const project = await plagiarismApi.createProject({ name: name.trim(), description: description.trim() || null });
      setName('');
      setDescription('');
      reload();
      onSelectProject(project);
    } catch (err) {
      console.error(err);
      alert((err as Error).message || '创建项目失败');
    } finally {
      setCreating(false);
    }
  };

  const handleNameChange = (event: ChangeEvent<HTMLInputElement>) => {
    setName(event.target.value);
  };

  const handleDescriptionChange = (event: ChangeEvent<HTMLInputElement>) => {
    setDescription(event.target.value);
  };

  const projectCount = projects?.length ?? 0;

  return (
    <PageShell>
      <PageHeader
        title="项目管理"
        subtitle="按照项目组织文件上传与比对，便于生成综合报告。"
        meta={loading ? '加载中…' : `项目数：${projectCount}`}
        actions={
          <Button variant="outline" size="sm" className="hover:bg-primary/10 hover:text-primary hover:border-primary" onClick={reload} disabled={loading}>
            刷新列表
          </Button>
        }
      />
      <PageContent>
        <SectionCard>
          <div className="flex flex-col gap-4">
            <div>
              <h2 className="text-base font-semibold leading-6 text-foreground">创建项目</h2>
              <p className="mt-1 text-sm text-muted-foreground">为不同批次或团队建立独立的项目，便于管理上传文档与后续任务。</p>
            </div>
            <form className="grid gap-4 md:grid-cols-2" onSubmit={handleSubmit}>
              <div className="space-y-2">
                <Label htmlFor="project-name" className="text-sm font-medium text-muted-foreground">
                  项目名称
                </Label>
                <Input
                  id="project-name"
                  value={name}
                  onChange={handleNameChange}
                  placeholder="例如：2024 届毕业论文批次"
                  className="h-9"
                />
              </div>
              <div className="space-y-2 md:col-span-2 lg:col-span-1">
                <Label htmlFor="project-description" className="text-sm font-medium text-muted-foreground">
                  项目描述（可选）
                </Label>
                <Input
                  id="project-description"
                  value={description}
                  onChange={handleDescriptionChange}
                  placeholder="补充项目备注，方便团队协作"
                  className="h-9"
                />
              </div>
              <div className="flex items-center gap-3 md:col-span-2">
                <Button type="submit" variant="default" disabled={creating}>
                  {creating ? '正在创建…' : '创建项目'}
                </Button>
                <Button type="button" variant="ghost" className="hover:bg-primary/10 hover:text-primary" onClick={() => {
                  setName('');
                  setDescription('');
                }}
                >
                  重置
                </Button>
              </div>
            </form>
          </div>
        </SectionCard>

        <SectionCard padding="none">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-3">
            <div className="flex items-center gap-3 text-sm font-medium text-muted-foreground">
              <span>项目列表</span>
              <span className="text-xs text-muted-foreground">{projectCount} 个项目</span>
            </div>
            {loading && <span className="text-xs text-muted-foreground">加载中…</span>}
          </div>
          <div className="max-h-[420px] overflow-auto">
            {error && <div className="px-5 py-4 text-sm text-destructive">加载项目列表失败：{error}</div>}
            {!loading && !error && projectCount === 0 && (
              <div className="px-5 py-6 text-sm text-muted-foreground">暂无项目，请先在上方创建。</div>
            )}
            {!loading && !error && projectCount > 0 && projects && (
              <table className={TABLE_BASE}>
                <thead className={TABLE_HEAD}>
                  <tr>
                    <th className="h-11 px-5 align-middle">项目名称</th>
                    <th className="h-11 px-5 align-middle">项目描述</th>
                    <th className="h-11 px-5 align-middle">创建时间</th>
                    <th className="h-11 px-5 align-middle">更新时间</th>
                  </tr>
                </thead>
                <tbody className={TABLE_BODY}>
                  {projects.map((project) => (
                    <tr
                      key={project.id}
                      className="cursor-pointer transition hover:bg-muted/60"
                      onClick={() => onSelectProject(project)}
                    >
                      <td className="px-5 py-3 text-sm font-medium">
                        {project.name || `项目 #${project.id}`}
                      </td>
                      <td className="px-5 py-3 text-xs text-muted-foreground">{project.description || '—'}</td>
                      <td className="px-5 py-3 text-xs text-muted-foreground">
                        {new Date(project.created_at).toLocaleString()}
                      </td>
                      <td className="px-5 py-3 text-xs text-muted-foreground">
                        {new Date(project.updated_at).toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </SectionCard>
      </PageContent>
    </PageShell>
  );
}
