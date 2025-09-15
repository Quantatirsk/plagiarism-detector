#!/usr/bin/env python3
"""
文本相似度对比工具 - Streamlit 版本
"""

import asyncio
import sys
import os
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.embedding import EmbeddingService


def cosine_similarity(v1, v2):
    """计算余弦相似度"""
    v1, v2 = np.array(v1), np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# 初始化 session state
if 'embedding_service' not in st.session_state:
    st.session_state.embedding_service = EmbeddingService()

if 'history' not in st.session_state:
    st.session_state.history = []


def main():
    """主函数"""
    st.set_page_config(
        page_title="文本相似度对比工具",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 文本相似度对比工具")
    st.markdown("---")

    # 创建两列输入区域
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 第一段文本")
        text1 = st.text_area(
            "输入或粘贴文本",
            height=200,
            key="text1",
            placeholder="在这里输入第一段文本..."
        )

    with col2:
        st.subheader("📝 第二段文本")
        text2 = st.text_area(
            "输入或粘贴文本",
            height=200,
            key="text2",
            placeholder="在这里输入第二段文本..."
        )

    # 分析按钮
    col_button, col_empty = st.columns([1, 3])
    with col_button:
        analyze_button = st.button("🔄 分析相似度", type="primary", use_container_width=True)

    # 结果显示区域
    if analyze_button:
        if not text1 or not text2:
            st.error("❌ 两段文本都不能为空！")
        else:
            with st.spinner("🔄 正在计算相似度..."):
                try:
                    # 异步运行embedding
                    async def compute_similarity():
                        vector1 = await st.session_state.embedding_service.embed_text(text1)
                        vector2 = await st.session_state.embedding_service.embed_text(text2)
                        return cosine_similarity(vector1, vector2)

                    similarity = asyncio.run(compute_similarity())

                    # 保存到历史记录
                    st.session_state.history.append({
                        'text1': text1[:50] + '...' if len(text1) > 50 else text1,
                        'text2': text2[:50] + '...' if len(text2) > 50 else text2,
                        'similarity': similarity
                    })

                    # 显示结果
                    st.markdown("---")
                    st.subheader("📊 分析结果")

                    # 创建结果显示列
                    result_col1, result_col2, result_col3 = st.columns(3)

                    with result_col1:
                        st.metric("相似度分数", f"{similarity:.4f}")

                    with result_col2:
                        st.metric("百分比", f"{similarity*100:.2f}%")

                    with result_col3:
                        if similarity >= 0.9:
                            st.success("高度相似")
                        elif similarity >= 0.7:
                            st.warning("中度相似")
                        else:
                            st.info("低相似度")

                    # 相似度进度条
                    st.progress(similarity)

                    # 相似度解释
                    st.markdown("### 📈 相似度解读")
                    if similarity >= 0.9:
                        st.markdown("**高度相似**: 两段文本内容几乎相同，可能存在抄袭或复制。")
                    elif similarity >= 0.7:
                        st.markdown("**中度相似**: 两段文本有较多相似内容，可能存在部分借鉴。")
                    elif similarity >= 0.5:
                        st.markdown("**轻度相似**: 两段文本有一些共同主题或词汇。")
                    else:
                        st.markdown("**低相似度**: 两段文本基本不相关。")

                except Exception as e:
                    st.error(f"❌ 计算失败: {e}")

    # 历史记录侧边栏
    with st.sidebar:
        st.header("📜 历史记录")
        if st.session_state.history:
            for i, record in enumerate(reversed(st.session_state.history[-10:])):  # 显示最近10条
                with st.expander(f"记录 {len(st.session_state.history) - i}"):
                    st.write(f"**文本1**: {record['text1']}")
                    st.write(f"**文本2**: {record['text2']}")
                    st.write(f"**相似度**: {record['similarity']:.4f} ({record['similarity']*100:.2f}%)")

            if st.button("🗑️ 清除历史"):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("暂无历史记录")

        # 使用说明
        st.markdown("---")
        st.header("📖 使用说明")
        st.markdown("""
        1. 在左右两个文本框中输入要对比的文本
        2. 点击"分析相似度"按钮
        3. 查看相似度分析结果

        **相似度阈值参考**:
        - ≥ 90%: 高度相似
        - 70-90%: 中度相似
        - 50-70%: 轻度相似
        - < 50%: 低相似度
        """)


if __name__ == "__main__":
    main()