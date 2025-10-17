import { useEffect, useMemo, useState } from 'react';
import type {
  CompareJobSummary,
  ComparePairSummary,
  ProjectSummary,
  DocumentSummary,
} from '@/api/plagiarismApi';
import { ProjectsPage } from '@/pages/ProjectsPage';
import { ProjectDetailPanel } from '@/pages/ProjectDetailPanel';
import { ProjectJobPanel } from '@/pages/ProjectJobPanel';
import { PlanComparePage } from '@/pages/PlanComparePage';
import { ReportsPage } from '@/pages/ReportsPage';
import { useDocuments, useJobPairs, usePairReport, useDocumentDetail } from '@/hooks/useData';
import { useCachedPairData } from '@/hooks/useCachedPairData';
import { Button } from '@/components/ui/button';

function App() {
  const [view, setView] = useState<'projects' | 'project' | 'job' | 'pair' | 'reports'>('projects');
  const [selectedProject, setSelectedProject] = useState<ProjectSummary | null>(null);
  const [selectedJob, setSelectedJob] = useState<CompareJobSummary | null>(null);
  const [selectedPair, setSelectedPair] = useState<ComparePairSummary | null>(null);
  const [reportMode, setReportMode] = useState<'project' | 'job' | 'document'>('project');

  const projectDocumentsState = useDocuments(selectedProject ? { projectId: selectedProject.id } : undefined);
  const projectDocuments = projectDocumentsState.data ?? [];
  const documentLookup = useMemo(() => {
    return projectDocuments.reduce<Record<number, DocumentSummary>>((acc, document) => {
      acc[document.id] = document;
      return acc;
    }, {});
  }, [projectDocuments]);

  const jobPairsState = useJobPairs(selectedJob?.id ?? null);
  const jobPairs = jobPairsState.data ?? [];

  const hasActivePairs = useMemo(
    () => jobPairs.some((pair) => pair.status === 'pending' || pair.status === 'running'),
    [jobPairs],
  );

  // Get adjacent pair IDs for pre-fetching
  const adjacentPairIds = useMemo(() => {
    if (!selectedPair || !jobPairs.length) return [];
    const currentIndex = jobPairs.findIndex(p => p.id === selectedPair.id);
    const adjacent: number[] = [];
    if (currentIndex > 0) adjacent.push(jobPairs[currentIndex - 1].id);
    if (currentIndex < jobPairs.length - 1) adjacent.push(jobPairs[currentIndex + 1].id);
    return adjacent;
  }, [selectedPair, jobPairs]);

  const cachedPairData = useCachedPairData(selectedPair?.id ?? null, adjacentPairIds);

  useEffect(() => {
    if (!selectedPair || !jobPairsState.data) {
      return;
    }
    const next = jobPairsState.data.find((pair) => pair.id === selectedPair.id);
    if (!next) {
      setSelectedPair(null);
      if (view === 'pair') {
        setView('job');
      }
      return;
    }
    if (next !== selectedPair) {
      setSelectedPair(next);
    }
  }, [jobPairsState.data, selectedPair, setView, view]);

  useEffect(() => {
    if (!selectedJob || (view !== 'job' && view !== 'pair')) {
      return;
    }
    if (!hasActivePairs) {
      return;
    }
    const timer = window.setInterval(() => {
      jobPairsState.reload();
    }, 5000);
    return () => {
      window.clearInterval(timer);
    };
  }, [hasActivePairs, jobPairsState.reload, selectedJob?.id, view]);

  if (view === 'projects') {
    return (
      <ProjectsPage
        onSelectProject={(project) => {
          setSelectedProject(project);
          setSelectedJob(null);
          setSelectedPair(null);
          setView('project');
        }}
      />
    );
  }

  if (view === 'project' && selectedProject) {
    return (
      <ProjectDetailPanel
        project={selectedProject}
        onBack={() => {
          setView('projects');
          setSelectedProject(null);
          setSelectedJob(null);
          setSelectedPair(null);
        }}
        onOpenJob={(job) => {
          setSelectedJob(job);
          setSelectedPair(null);
          setView('job');
        }}
      />
    );
  }

  if (view === 'job' && selectedProject && selectedJob) {
    return (
      <ProjectJobPanel
        project={selectedProject}
        job={selectedJob}
        onBack={() => {
          setSelectedPair(null);
          setView('project');
        }}
        onOpenPair={(pair) => {
          setSelectedPair(pair);
          setView('pair');
        }}
        pairs={jobPairs}
        pairsLoading={jobPairsState.loading}
        pairsError={jobPairsState.error}
        onReloadPairs={jobPairsState.reload}
        documentLookup={documentLookup}
        onOpenReports={() => {
          setReportMode('job');
          setView('reports');
        }}
      />
    );
  }

  if (view === 'pair' && selectedPair && selectedProject && selectedJob) {
    if (cachedPairData.loading) {
      return (
        <PlanComparePage
          report={cachedPairData.report}
          leftDocument={cachedPairData.leftDocument}
          rightDocument={cachedPairData.rightDocument}
          pairs={jobPairs}
          pairsLoading={jobPairsState.loading}
          pairsError={jobPairsState.error}
        documentLookup={documentLookup}
        onSwitchPair={(pairId) => {
          const next = jobPairs.find((pair) => pair.id === pairId);
          if (next) {
            setSelectedPair(next);
          }
        }}
        onBack={() => setView('job')}
        isTransitioning={true}
      />
      );
    }

    if (cachedPairData.error || !cachedPairData.report) {
      return (
        <div className="flex h-full flex-col items-center justify-center gap-3 bg-muted/40 text-sm text-destructive">
          加载配对报告失败：{cachedPairData.error ?? '未知错误'}
          <Button variant="outline" size="sm" onClick={() => setView('job')}>
            返回任务
          </Button>
        </div>
      );
    }

    return (
      <PlanComparePage
        report={cachedPairData.report}
        leftDocument={cachedPairData.leftDocument}
        rightDocument={cachedPairData.rightDocument}
        pairs={jobPairs}
        pairsLoading={jobPairsState.loading}
        pairsError={jobPairsState.error}
        documentLookup={documentLookup}
        onSwitchPair={(pairId) => {
          const next = jobPairs.find((pair) => pair.id === pairId);
          if (next) {
            setSelectedPair(next);
          }
        }}
        onBack={() => setView('job')}
        isTransitioning={false}
      />
    );
  }

  // Reports view
  if (view === 'reports') {
    // Prepare available documents and projects data
    const sortedDocuments = [...projectDocuments].sort((a, b) => {
      const aTime = new Date(a.created_at).getTime();
      const bTime = new Date(b.created_at).getTime();
      return aTime - bTime;
    });

    const availableDocuments = sortedDocuments.map(doc => ({
      id: doc.id.toString(),
      title: doc.title || doc.filename || `文档 ${doc.id}`
    }));

    const availableProjects = selectedProject ? [{
      id: selectedProject.id.toString(),
      name: selectedProject.name
    }] : [];

    // Determine navigation context based on report mode
    const handleBack = () => {
      if (reportMode === 'project' && selectedProject) {
        setView('project');
      } else if (reportMode === 'job' && selectedJob) {
        setView('job');
      } else {
        setView('projects');
      }
    };

    return (
      <ReportsPage
        mode={reportMode}
        project={selectedProject || undefined}
        job={selectedJob || undefined}
        availableDocuments={availableDocuments}
        availableProjects={availableProjects}
        onBack={handleBack}
      />
    );
  }

  return null;
}

export default App;
