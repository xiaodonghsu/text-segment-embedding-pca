import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import time

# 页面配置
st.set_page_config(
    page_title="RAG可视化教学系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RAGVisualizer:
    def __init__(self):
        if 'segments' not in st.session_state:
            st.session_state.segments = []
        if 'embeddings' not in st.session_state:
            st.session_state.embeddings = []
        if 'pca_data' not in st.session_state:
            st.session_state.pca_data = None
        if 'pca_model' not in st.session_state:
            st.session_state.pca_model = None
        if 'similarity_matrix' not in st.session_state:
            st.session_state.similarity_matrix = None
        if 'query_embedding' not in st.session_state:
            st.session_state.query_embedding = None
        if 'query_pca' not in st.session_state:
            st.session_state.query_pca = None

    def clear_all_data(self):
        """清除所有数据"""
        st.session_state.segments = []
        st.session_state.embeddings = []
        st.session_state.pca_data = None
        st.session_state.pca_model = None
        st.session_state.similarity_matrix = None
        st.session_state.query_embedding = None
        st.session_state.query_pca = None

    def split_text(self, text, chunk_size, overlap_size):
        """文本分段"""
        segments = []
        start = 0
        segment_id = 1
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            segment = {
                'id': segment_id,
                'content': text[start:end],
                'start': start,
                'end': end,
                'length': end - start
            }
            segments.append(segment)
            segment_id += 1
            
            if end >= len(text):
                break
            start = end - overlap_size
        
        return segments

    def call_embedding_api(self, text, api_config):
        """调用embedding API"""
        if not all([api_config['base_url'], api_config['api_key'], api_config['model']]):
            raise Exception('请完整配置API参数（Base URL、API Key、模型名称）')
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_config["api_key"]}'
        }
        
        data = {
            'model': api_config['model'],
            'input': text,
            'encoding_format': 'float'
        }
        
        response = requests.post(api_config['base_url'], headers=headers, json=data)
        
        if response.status_code != 200:
            raise Exception(f'API调用失败: {response.status_code} {response.text}')
        
        result = response.json()
        
        if not result.get('data') or not result['data'][0].get('embedding'):
            raise Exception('API返回数据格式错误')
        
        return result['data'][0]['embedding']

    def generate_embeddings(self, segments, api_config):
        """批量生成embeddings"""
        embeddings = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, segment in enumerate(segments):
            status_text.text(f'正在处理分段 {i + 1}/{len(segments)}...')
            
            try:
                embedding = self.call_embedding_api(segment['content'], api_config)
                embeddings.append(embedding)
                progress_bar.progress((i + 1) / len(segments))
                time.sleep(0.1)  # 避免API限流
            except Exception as e:
                st.error(f'处理分段 {i + 1} 时出错: {str(e)}')
                return None
        
        status_text.text('向量生成完成！')
        return embeddings

    def perform_pca(self, embeddings):
        """执行PCA降维"""
        pca = PCA(n_components=2, random_state=42)
        pca_data = pca.fit_transform(embeddings)
        return pca_data, pca

    def calculate_similarity_matrix(self, embeddings):
        """计算相似度矩阵"""
        return cosine_similarity(embeddings)

    def create_pca_plot(self, pca_data, segments, query_pca=None):
        """创建PCA可视化图"""
        fig = go.Figure()
        
        # 添加文本分段点
        fig.add_trace(go.Scatter(
            x=pca_data[:, 0],
            y=pca_data[:, 1],
            mode='markers+text',
            text=[f'分段{s["id"]}' for s in segments],
            textposition='top center',
            marker=dict(
                size=12,
                color='#667eea',
                line=dict(width=2, color='#4c51bf')
            ),
            name='文本分段',
            hovertemplate='<b>分段%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
        ))
        
        # 如果有查询向量，也显示
        if query_pca is not None:
            fig.add_trace(go.Scatter(
                x=[query_pca[0]],
                y=[query_pca[1]],
                mode='markers+text',
                text=['查询'],
                textposition='top center',
                marker=dict(
                    size=15,
                    color='#e53e3e',
                    symbol='star',
                    line=dict(width=2, color='#c53030')
                ),
                name='查询问题',
                hovertemplate='<b>查询问题</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='文本分段向量PCA降维可视化',
            xaxis_title='第一主成分',
            yaxis_title='第二主成分',
            showlegend=True,
            height=500,
            template='plotly_white'
        )
        
        return fig

    def create_similarity_heatmap(self, similarity_matrix, segments):
        """创建相似度热力图"""
        labels = [f'分段{s["id"]}' for s in segments]
        
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=np.round(similarity_matrix * 100, 1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            hovertemplate='分段%{y} vs 分段%{x}<br>相似度: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='分段间相似度矩阵',
            height=400,
            template='plotly_white'
        )
        
        return fig

def main():
    st.title("🤖 RAG可视化教学系统")
    st.markdown("体验文本分段、向量化、PCA降维和相似度计算的完整流程")
    
    visualizer = RAGVisualizer()
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 系统配置")
        
        # API配置区域
        st.subheader("🔑 Embedding API配置")
        
        api_base_url = st.text_input(
            "Base URL",
            value="",
            placeholder="例如: https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
            help="Embedding API的基础URL地址"
        )
        
        api_key = st.text_input(
            "API Key",
            value="",
            type="password",
            placeholder="请输入您的API密钥",
            help="您的API密钥"
        )
        
        api_model = st.text_input(
            "模型名称",
            value="",
            placeholder="例如: text-embedding-v4",
            help="使用的embedding模型名称"
        )
        
        # 构建API配置
        api_config = {
            'base_url': api_base_url.strip(),
            'api_key': api_key.strip(),
            'model': api_model.strip(),
            'dimension': 1024
        }
        
        st.divider()
        
        # 分段参数配置
        st.subheader("📝 分段参数")
        chunk_size = st.selectbox(
            "分段长度",
            options=[200, 500],
            index=1,
            help="选择文本分段的字符长度"
        )
        
        overlap_percent = st.slider(
            "重叠度 (%)",
            min_value=0,
            max_value=100,
            value=50,
            help="相邻分段之间的重叠百分比"
        )
        
        st.divider()
        
        # 数据管理
        st.subheader("🗂️ 数据管理")
        if st.button("🗑️ 清除所有数据", type="secondary"):
            visualizer.clear_all_data()
            st.success("✅ 所有数据已清除！")
            st.rerun()
        
        st.divider()
        
        # 系统状态
        st.subheader("📊 系统状态")
        
        # API配置状态检查
        api_configured = all([
            api_config['base_url'],
            api_config['api_key'],
            api_config['model']
        ])
        
        if api_configured:
            st.success("✅ API配置完整")
        else:
            missing_items = []
            if not api_config['base_url']:
                missing_items.append("Base URL")
            if not api_config['api_key']:
                missing_items.append("API Key")
            if not api_config['model']:
                missing_items.append("模型名称")
            st.warning(f"⚠️ 请配置: {', '.join(missing_items)}")
            
        if st.session_state.segments:
            st.success(f"✅ 已分段: {len(st.session_state.segments)} 个")
        if st.session_state.embeddings:
            st.success(f"✅ 已向量化: {len(st.session_state.embeddings)} 个")
        if st.session_state.pca_data is not None:
            st.success("✅ PCA降维完成")
    
    # 主要内容区域
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 文本输入", "🔍 分段结果", "📊 向量可视化", "❓ 问题查询", "📈 相似度矩阵"
    ])
    
    with tab1:
        st.header("1. 文本输入与分段")
        
        # 默认文本
        default_text = """人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。

机器学习是人工智能的一个重要分支，它是一种通过算法使机器能够从数据中学习并做出决策或预测的技术。机器学习算法通过训练数据来识别模式，然后使用这些模式来对新数据进行预测或分类。

深度学习是机器学习的一个子集，它使用人工神经网络来模拟人脑的工作方式。深度学习在图像识别、语音识别、自然语言处理等领域取得了突破性进展。神经网络由多个层组成，每一层都包含多个神经元，这些神经元通过权重连接。

自然语言处理（Natural Language Processing，NLP）是人工智能和语言学领域的分支学科。它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。NLP是计算机科学领域与人工智能领域中的一个重要方向。

计算机视觉是一门研究如何使机器"看"的科学，更进一步的说，就是指用摄影机和电脑代替人眼对目标进行识别、跟踪和测量等机器视觉，并进一步做图形处理，使电脑处理成为更适合人眼观察或传送给仪器检测的图像。

强化学习是机器学习中的一个领域，强调如何基于环境而行动，以取得最大化的预期利益。强化学习是除了监督学习和无监督学习之外的第三种基本的机器学习方法。与监督学习和无监督学习不同，强化学习不需要出现正确的输入/输出对，也不需要精确校正次优化的行为。"""
        
        input_text = st.text_area(
            "输入长文本内容",
            value=default_text,
            height=200,
            help="输入需要进行RAG处理的长文本"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 处理文本", type="primary"):
                if input_text.strip():
                    overlap_size = int(chunk_size * overlap_percent / 100)
                    segments = visualizer.split_text(input_text.strip(), chunk_size, overlap_size)
                    
                    st.session_state.segments = segments
                    # 清除之前的向量化数据
                    st.session_state.embeddings = []
                    st.session_state.pca_data = None
                    st.session_state.pca_model = None
                    st.session_state.similarity_matrix = None
                    st.session_state.query_embedding = None
                    st.session_state.query_pca = None
                    
                    st.success(f"✅ 文本处理完成！共生成 {len(segments)} 个分段")
                    st.rerun()
                else:
                    st.error("请输入文本内容")
    
    with tab2:
        st.header("2. 文本分段结果")
        
        if st.session_state.segments:
            st.info(f"📊 共生成 {len(st.session_state.segments)} 个分段，重叠度 {overlap_percent}%")
            
            for segment in st.session_state.segments:
                with st.expander(f"分段 {segment['id']} ({segment['length']} 字符)"):
                    st.write(segment['content'])
        else:
            st.warning("请先在「文本输入」标签页处理文本")
    
    with tab3:
        st.header("3. 向量化与PCA降维可视化")
        
        if st.session_state.segments:
            if not api_configured:
                st.error("⚠️ 请先在侧边栏完整配置API参数（Base URL、API Key、模型名称）")
            else:
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("🚀 生成向量并可视化", type="primary"):
                        with st.spinner("正在调用embedding API..."):
                            embeddings = visualizer.generate_embeddings(st.session_state.segments, api_config)
                            
                            if embeddings:
                                st.session_state.embeddings = embeddings
                                
                                # 执行PCA降维
                                pca_data, pca_model = visualizer.perform_pca(embeddings)
                                st.session_state.pca_data = pca_data
                                st.session_state.pca_model = pca_model
                                
                                # 计算相似度矩阵
                                similarity_matrix = visualizer.calculate_similarity_matrix(embeddings)
                                st.session_state.similarity_matrix = similarity_matrix
                                
                                st.success("✅ 向量生成和PCA降维完成！")
                                st.rerun()
                
                if st.session_state.pca_data is not None:
                    st.subheader("📊 PCA降维可视化")
                    fig = visualizer.create_pca_plot(
                        st.session_state.pca_data, 
                        st.session_state.segments,
                        st.session_state.query_pca
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示PCA解释方差比
                    explained_variance = st.session_state.pca_model.explained_variance_ratio_
                    st.info(f"📈 PCA解释方差比: PC1={explained_variance[0]:.3f}, PC2={explained_variance[1]:.3f}")
        else:
            st.warning("请先在「文本输入」标签页处理文本")
    
    with tab4:
        st.header("4. 问题查询")
        
        if st.session_state.embeddings:
            if not api_configured:
                st.error("⚠️ 请先在侧边栏完整配置API参数（Base URL、API Key、模型名称）")
            else:
                query = st.text_input(
                    "输入您的问题",
                    placeholder="例如：什么是深度学习？",
                    help="输入问题，系统会找到最相关的文本分段"
                )
                
                if st.button("🔍 查询", type="primary") and query.strip():
                    with st.spinner("正在处理查询..."):
                        try:
                            # 生成查询向量
                            query_embedding = visualizer.call_embedding_api(query.strip(), api_config)
                            st.session_state.query_embedding = query_embedding
                            
                            # PCA投影
                            query_pca = st.session_state.pca_model.transform([query_embedding])[0]
                            st.session_state.query_pca = query_pca
                            
                            # 计算相似度
                            similarities = cosine_similarity([query_embedding], st.session_state.embeddings)[0]
                            
                            # 排序并显示结果
                            results = []
                            for i, sim in enumerate(similarities):
                                results.append({
                                    'segment_id': st.session_state.segments[i]['id'],
                                    'similarity': sim,
                                    'content': st.session_state.segments[i]['content']
                                })
                            
                            results.sort(key=lambda x: x['similarity'], reverse=True)
                            
                            st.success("✅ 查询完成！")
                            
                            # 显示查询结果
                            st.subheader(f"🎯 查询问题：{query}")
                            st.subheader("📋 最相关的文本分段：")
                            
                            for i, result in enumerate(results[:3]):
                                similarity_percent = result['similarity'] * 100
                                
                                if i == 0:
                                    st.success(f"🥇 **分段 {result['segment_id']}** (相似度: {similarity_percent:.1f}%)")
                                elif i == 1:
                                    st.info(f"🥈 **分段 {result['segment_id']}** (相似度: {similarity_percent:.1f}%)")
                                else:
                                    st.warning(f"🥉 **分段 {result['segment_id']}** (相似度: {similarity_percent:.1f}%)")
                                
                                content_preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                                st.write(content_preview)
                                st.divider()
                            
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"查询处理失败: {str(e)}")
        else:
            st.warning("请先在「向量可视化」标签页生成向量")
    
    with tab5:
        st.header("5. 分段相似度矩阵")
        
        if st.session_state.similarity_matrix is not None:
            st.subheader("🔥 相似度热力图")
            fig = visualizer.create_similarity_heatmap(
                st.session_state.similarity_matrix,
                st.session_state.segments
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示数值矩阵
            st.subheader("📊 相似度数值矩阵")
            df = pd.DataFrame(
                st.session_state.similarity_matrix,
                columns=[f'分段{s["id"]}' for s in st.session_state.segments],
                index=[f'分段{s["id"]}' for s in st.session_state.segments]
            )
            st.dataframe(df.round(3), use_container_width=True)
        else:
            st.warning("请先在「向量可视化」标签页生成向量")

if __name__ == "__main__":
    main()