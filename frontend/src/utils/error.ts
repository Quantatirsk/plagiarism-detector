/**
 * 错误处理工具函数
 */

export type ErrorType = 'auth' | 'not-found' | 'network' | 'validation' | 'unknown';

export interface ErrorConfig {
  icon: string;
  title: string;
  description: string;
  actionText: string;
}

/**
 * 获取错误类型
 */
export function getErrorType(error: unknown): ErrorType {
  if (!error) return 'unknown';

  // 检查是否是字符串
  if (typeof error === 'string') {
    if (error.includes('401') || error.includes('unauthorized')) return 'auth';
    if (error.includes('404') || error.includes('not found')) return 'not-found';
    if (error.includes('network') || error.includes('timeout')) return 'network';
    return 'unknown';
  }

  // 检查是否是 Error 对象
  if (error instanceof Error) {
    const message = error.message.toLowerCase();
    if (message.includes('401') || message.includes('unauthorized')) return 'auth';
    if (message.includes('404') || message.includes('not found')) return 'not-found';
    if (message.includes('network') || message.includes('timeout')) return 'network';
    if (message.includes('validation') || message.includes('invalid')) return 'validation';
    return 'unknown';
  }

  // 检查是否是 Axios 错误
  if (typeof error === 'object' && error !== null) {
    const errorObj = error as any;
    if (errorObj.response) {
      const status = errorObj.response.status;
      if (status === 401 || status === 403) return 'auth';
      if (status === 404) return 'not-found';
      if (status >= 400 && status < 500) return 'validation';
    }
    if (errorObj.request) return 'network';
  }

  return 'unknown';
}

/**
 * 获取错误配置
 */
export function getErrorConfig(type: ErrorType, customMessage?: string): ErrorConfig {
  const configs: Record<ErrorType, ErrorConfig> = {
    auth: {
      icon: '🔒',
      title: '未授权',
      description: customMessage || '您没有权限访问此资源，请重新登录',
      actionText: '重新登录',
    },
    'not-found': {
      icon: '🔍',
      title: '资源不存在',
      description: customMessage || '您访问的资源不存在或已被删除',
      actionText: '返回首页',
    },
    network: {
      icon: '📡',
      title: '网络错误',
      description: customMessage || '网络连接失败，请检查您的网络设置',
      actionText: '重试',
    },
    validation: {
      icon: '⚠️',
      title: '验证失败',
      description: customMessage || '请求参数不正确，请检查输入',
      actionText: '重试',
    },
    unknown: {
      icon: '❌',
      title: '发生错误',
      description: customMessage || '发生未知错误，请稍后重试',
      actionText: '重试',
    },
  };

  return configs[type];
}
