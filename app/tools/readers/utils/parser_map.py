"""
解析器映射配置

单一职责：定义文件扩展名到解析器的映射关系
符合 Linux 哲学：配置与代码分离
"""

# 专用解析器映射（优先级高）
SPECIALIZED_PARSERS = {
    # 文档格式 - 有专门的二进制解析器
    '.pdf': ('app.tools.readers.readers_pdf', 'PDFParser'),
    '.docx': ('app.tools.readers.readers_docx', 'DOCXParser'),
    '.doc': ('app.tools.readers.readers_doc', 'DOCParser'),
}

# 文本解析器支持的格式
# 使用 EnhancedTextParser 处理所有基于文本的格式
TEXT_PARSER_FORMATS = [
    # ========== 基本文本文件 ==========
    '.txt', '.text', '.md', '.markdown', '.rst', '.rtf',
    
    # ========== 编程语言 ==========
    # Python
    '.py', '.pyx', '.pyi', '.pyw',
    # JavaScript/TypeScript
    '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs',
    # JVM languages
    '.java', '.scala', '.kotlin', '.groovy', '.clj', '.cljs', '.cljc', '.edn',
    # C/C++
    '.c', '.h', '.cpp', '.cxx', '.cc', '.hpp', '.hxx', '.hh',
    # .NET languages
    '.cs', '.vb', '.fs', '.fsx',
    # PHP
    '.php', '.php3', '.php4', '.php5', '.phtml',
    # Ruby
    '.rb', '.rbw', '.rake', '.gemspec',
    # Go
    '.go', '.mod', '.sum',
    # Rust
    '.rs', '.toml',
    # Swift
    '.swift',
    # Dart/Flutter
    '.dart',
    # Kotlin
    '.kt', '.kts',
    # Perl
    '.pl', '.pm', '.pod', '.t',
    # Lua
    '.lua',
    # R
    '.r', '.R', '.rmd', '.Rmd',
    # Objective-C/C++
    '.m', '.mm',
    # Pascal/Delphi
    '.pas', '.pp', '.inc', '.dpr', '.dpk',
    # Assembly
    '.asm', '.s', '.S',
    # SQL
    '.sql', '.mysql', '.pgsql', '.sqlite', '.plsql',
    # Shell/Batch
    '.sh', '.bash', '.zsh', '.fish', '.csh', '.tcsh', '.ksh',
    '.bat', '.cmd', '.ps1', '.psm1', '.psd1',
    # Other languages
    '.vbs', '.vba',  # Visual Basic
    '.nim', '.nims',  # Nim
    '.zig',  # Zig
    '.jl',  # Julia
    '.elm',  # Elm
    '.ex', '.exs',  # Elixir
    '.erl', '.hrl',  # Erlang
    '.hs', '.lhs',  # Haskell
    '.ml', '.mli',  # OCaml
    '.v', '.vh', '.sv', '.svh',  # Verilog/SystemVerilog
    '.vhd', '.vhdl',  # VHDL
    '.tcl',  # Tcl
    '.lisp', '.lsp', '.cl', '.el',  # Lisp dialects
    '.scm', '.ss', '.rkt',  # Scheme/Racket
    '.f', '.f90', '.f95', '.f03', '.f08', '.for',  # Fortran
    '.cob', '.cbl', '.cpy',  # COBOL
    '.ada', '.adb', '.ads',  # Ada
    '.d', '.di',  # D
    '.cr',  # Crystal
    '.hx', '.hxml',  # Haxe
    '.purs',  # PureScript
    '.reason', '.re', '.rei',  # ReasonML
    '.coffee',  # CoffeeScript
    '.ls',  # LiveScript
    '.flow',  # Flow
    '.ino', '.pde',  # Arduino
    '.proto',  # Protocol Buffers
    '.thrift',  # Apache Thrift
    '.graphql', '.gql',  # GraphQL
    
    # ========== Web 技术 ==========
    '.html', '.htm', '.xhtml', '.shtml',
    '.css', '.scss', '.sass', '.less', '.styl',
    '.vue', '.svelte',
    '.xml', '.xsl', '.xslt', '.xsd', '.dtd',
    '.svg',
    '.jsp', '.jspx', '.asp', '.aspx',
    '.ejs', '.erb', '.haml', '.jade', '.pug',
    '.mustache', '.hbs', '.handlebars',
    '.twig', '.liquid',
    
    # ========== 配置文件 ==========
    '.json', '.jsonc', '.json5',
    '.yaml', '.yml',
    '.toml',
    '.ini', '.cfg', '.conf', '.config',
    '.properties',
    '.env', '.environment',
    '.dockerfile', '.containerfile',
    '.makefile', '.mk', '.mak',
    '.cmake',
    '.gradle', '.gradle.kts',
    '.sbt',
    '.pom',
    '.build', '.bazel', '.bzl',
    '.nix',
    '.terraform', '.tf', '.tfvars', '.hcl',
    '.k8s', '.kube',
    '.ansible',
    '.vagrant', '.vagrantfile',
    '.jenkinsfile',
    '.travis.yml', '.circleci', '.github',
    '.gitlab-ci.yml',
    '.npmrc', '.yarnrc', '.nvmrc',
    '.gemrc', '.rvmrc', '.ruby-version',
    '.pythonrc', '.pypirc',
    '.cargo', '.rustfmt.toml',
    
    # ========== 构建文件 ==========
    '.make', '.am', '.in',
    '.pro', '.pri', '.qml',
    '.vcxproj', '.vcproj', '.sln', '.csproj', '.vbproj', '.fsproj',
    '.pbxproj', '.xcodeproj', '.xcworkspace',
    '.workspace', '.project',
    
    # ========== 文档格式 ==========
    '.tex', '.latex', '.cls', '.sty', '.bib',
    '.pod',
    '.rdoc',
    '.org',
    '.wiki',
    '.textile',
    '.asciidoc', '.adoc', '.asc',
    '.pandoc',
    
    # ========== 数据格式 ==========
    '.tsv', '.tab',
    '.log',
    '.diff', '.patch',
    '.gitignore', '.gitattributes', '.gitmodules', '.gitconfig',
    '.hgignore', '.hgrc',
    '.svnignore',
    '.editorconfig',
    '.eslintrc', '.eslintignore',
    '.prettierrc', '.prettierignore',
    '.babelrc',
    '.pylintrc', '.flake8', '.mypy.ini',
    '.rubocop.yml',
    '.stylelintrc',
    '.markdownlint.json',
    
    # ========== 其他文本格式 ==========
    '.readme', '.license', '.changelog', '.authors', '.contributors',
    '.todo', '.fixme',
    '.spec', '.test',
    '.template', '.tmpl', '.tpl',
    '.snippet', '.snip',
    '.example', '.sample',
    '.manifest',
    '.version',
    '.lock',
    '.sum',
    '.sig', '.asc', '.gpg',
    '.pem', '.crt', '.key', '.pub',
]

# 无扩展名文件（通常也是文本）
EXTENSIONLESS_TEXT_FILES = [
    'dockerfile', 'containerfile',
    'makefile', 'gnumakefile',
    'rakefile', 'gemfile', 'guardfile', 'capfile', 'thorfile',
    'vagrantfile', 'berksfile', 'puppetfile',
    'procfile', 'appfile',
    'buildfile', 'workspace',
    'license', 'licence', 'copyright', 'copying',
    'readme', 'changelog', 'history', 'changes',
    'authors', 'contributors', 'acknowledgments', 'credits',
    'install', 'installation',
    'news', 'releases',
    'todo', 'tasks', 'bugs', 'issues',
    'roadmap', 'milestones',
    'faq', 'questions',
    'contributing', 'code_of_conduct',
    'security',
    'notice', 'disclaimer',
    'manifest',
    'version',
    'config', 'configure',
    'setup',
    'requirements',
    'dependencies',
]

def get_parser_map():
    """
    获取完整的解析器映射
    
    Returns:
        dict: 文件扩展名到解析器的映射
    """
    parser_map = {}
    
    # 1. 先添加文本解析器（默认）
    for ext in TEXT_PARSER_FORMATS:
        parser_map[ext.lower()] = ('app.tools.readers.readers_text', 'EnhancedTextParser')
    
    # 2. 覆盖专用解析器（优先级更高）
    parser_map.update(SPECIALIZED_PARSERS)
    
    return parser_map

def is_extensionless_text_file(filename):
    """
    检查是否是无扩展名的文本文件
    
    Args:
        filename: 文件名（不含路径）
        
    Returns:
        bool: 是否是已知的无扩展名文本文件
    """
    return filename.lower() in EXTENSIONLESS_TEXT_FILES

def get_parser_for_file(file_path):
    """
    根据文件路径获取合适的解析器
    
    Args:
        file_path: 文件路径
        
    Returns:
        tuple: (module_path, class_name) 或 None
    """
    from pathlib import Path
    
    path = Path(file_path)
    ext = path.suffix.lower()
    name = path.name.lower()
    
    # 1. 先检查扩展名
    parser_map = get_parser_map()
    if ext in parser_map:
        return parser_map[ext]
    
    # 2. 检查无扩展名文件
    if not ext and is_extensionless_text_file(name):
        return ('app.tools.readers.readers_text', 'EnhancedTextParser')
    
    return None

# 统计信息
def get_format_stats():
    """获取格式支持的统计信息"""
    parser_map = get_parser_map()
    
    stats = {
        'total': len(parser_map),
        'text_formats': len(TEXT_PARSER_FORMATS),
        'specialized_formats': len(SPECIALIZED_PARSERS),
        'extensionless_files': len(EXTENSIONLESS_TEXT_FILES),
    }
    
    # 按解析器分组
    by_parser = {}
    for ext, (module, parser) in parser_map.items():
        key = f"{module}.{parser}"
        if key not in by_parser:
            by_parser[key] = []
        by_parser[key].append(ext)
    
    stats['by_parser'] = {k: len(v) for k, v in by_parser.items()}
    
    return stats