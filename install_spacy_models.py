#!/usr/bin/env python3
"""
安装 spaCy 模型
"""
import subprocess
import sys

def install_spacy_models():
    """安装中英文 spaCy 模型"""
    models = [
        "zh_core_web_md",  # 中文小模型
        "en_core_web_md",  # 英文小模型
    ]

    for model in models:
        print(f"正在安装 {model}...")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
            print(f"✅ {model} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ {model} 安装失败: {e}")
            print(f"请手动运行: python -m spacy download {model}")

if __name__ == "__main__":
    print("开始安装 spaCy 模型...")
    install_spacy_models()
    print("\n安装完成！")