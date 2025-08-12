class RAGVisualization {
    constructor() {
        this.segments = [];
        this.embeddings = [];
        this.pcaData = null;
        this.queryEmbedding = null;
        this.similarityMatrix = null;
        this.pcaTransform = null;
        
        // API配置 - 需要用户自行配置
        this.apiConfig = {
            baseUrl: '',
            apiKey: '',
            model: '',
            dimension: 1024
        };
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        document.getElementById('processText').addEventListener('click', () => this.processText());
        document.getElementById('generateEmbeddings').addEventListener('click', () => this.generateEmbeddings());
        document.getElementById('queryButton').addEventListener('click', () => this.processQuery());
        
        // 回车键查询
        document.getElementById('queryInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.processQuery();
            }
        });
    }

    // 文本分段
    processText() {
        const text = document.getElementById('inputText').value.trim();
        if (!text) {
            alert('请输入文本内容');
            return;
        }

        const chunkSize = parseInt(document.getElementById('chunkSize').value);
        const overlapPercent = parseInt(document.getElementById('overlap').value);
        const overlapSize = Math.floor(chunkSize * overlapPercent / 100);

        this.segments = this.splitText(text, chunkSize, overlapSize);
        this.displaySegments();
        
        // 重置其他数据
        this.embeddings = [];
        this.pcaData = null;
        this.queryEmbedding = null;
        this.pcaTransform = null;
        this.clearVisualization();
    }

    splitText(text, chunkSize, overlapSize) {
        const segments = [];
        let start = 0;
        let segmentId = 1;

        while (start < text.length) {
            const end = Math.min(start + chunkSize, text.length);
            const segment = {
                id: segmentId++,
                content: text.slice(start, end),
                start: start,
                end: end
            };
            segments.push(segment);

            if (end >= text.length) break;
            start = end - overlapSize;
        }

        return segments;
    }

    displaySegments() {
        const container = document.getElementById('segments');
        container.innerHTML = '';

        this.segments.forEach(segment => {
            const segmentDiv = document.createElement('div');
            segmentDiv.className = 'segment';
            segmentDiv.innerHTML = `
                <div class="segment-header">
                    <span class="segment-id">分段 ${segment.id}</span>
                    <span class="segment-length">${segment.content.length} 字符</span>
                </div>
                <div class="segment-content">${segment.content}</div>
            `;
            container.appendChild(segmentDiv);
        });
    }

    // 调用真实的embedding API
    async generateEmbeddings() {
        if (this.segments.length === 0) {
            alert('请先处理文本分段');
            return;
        }

        const button = document.getElementById('generateEmbeddings');
        const status = document.getElementById('embeddingStatus');
        
        button.disabled = true;
        status.className = 'status loading';
        status.innerHTML = '<span class="loading-spinner"></span> 正在调用embedding API...';

        try {
            // 批量调用embedding API
            this.embeddings = [];
            
            for (let i = 0; i < this.segments.length; i++) {
                status.innerHTML = `<span class="loading-spinner"></span> 正在处理分段 ${i + 1}/${this.segments.length}...`;
                
                const embedding = await this.callEmbeddingAPI(this.segments[i].content);
                this.embeddings.push(embedding);
                
                // 添加小延迟避免API限流
                await this.delay(200);
            }
            
            // 执行PCA降维
            this.performPCA();
            
            // 计算相似度矩阵
            this.calculateSimilarityMatrix();
            
            // 可视化
            this.visualizePCA();
            
            status.className = 'status success';
            status.textContent = `向量生成完成！共处理 ${this.segments.length} 个分段`;
            
        } catch (error) {
            console.error('Embedding API调用失败:', error);
            status.className = 'status error';
            status.textContent = '调用embedding API时出错：' + error.message;
        } finally {
            button.disabled = false;
        }
    }

    // 调用embedding API
    async callEmbeddingAPI(text) {
        const response = await fetch(this.apiConfig.baseUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.apiConfig.apiKey}`
            },
            body: JSON.stringify({
                model: this.apiConfig.model,
                input: text,
                encoding_format: 'float'
            })
        });

        if (!response.ok) {
            throw new Error(`API调用失败: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        
        if (!data.data || !data.data[0] || !data.data[0].embedding) {
            throw new Error('API返回数据格式错误');
        }

        return data.data[0].embedding;
    }

    // PCA降维
    performPCA() {
        if (this.embeddings.length === 0) return;

        try {
            // 检查MLMatrix是否可用
            if (typeof MLMatrix !== 'undefined' && MLMatrix.Matrix) {
                this.performPCAWithMLMatrix();
            } else {
                // 使用简化的PCA实现
                this.performSimplifiedPCA();
            }
        } catch (error) {
            console.warn('MLMatrix PCA失败，使用简化版本:', error);
            this.performSimplifiedPCA();
        }
    }

    // 使用MLMatrix的PCA实现
    performPCAWithMLMatrix() {
        const matrix = new MLMatrix.Matrix(this.embeddings);
        
        // 中心化
        const mean = matrix.mean('column');
        const centeredMatrix = matrix.subRowVector(mean);
        
        // 计算协方差矩阵
        const covariance = centeredMatrix.transpose().mmul(centeredMatrix).div(centeredMatrix.rows - 1);
        
        // 特征值分解
        const { eigenvectors } = new MLMatrix.EigenvalueDecomposition(covariance);
        
        // 取前两个主成分
        const pc1 = eigenvectors.getColumn(eigenvectors.columns - 1);
        const pc2 = eigenvectors.getColumn(eigenvectors.columns - 2);
        const principalComponents = new MLMatrix.Matrix([pc1, pc2]).transpose();
        
        // 保存PCA变换参数
        this.pcaTransform = {
            mean: mean.to1DArray(),
            components: principalComponents.to2DArray()
        };
        
        // 投影到2D空间
        const projected = centeredMatrix.mmul(principalComponents);
        this.pcaData = {
            getColumn: (i) => projected.getColumn(i)
        };
    }

    // 简化的PCA实现
    performSimplifiedPCA() {
        const n = this.embeddings.length;
        const d = this.embeddings[0].length;
        
        // 计算均值
        const mean = new Array(d).fill(0);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < d; j++) {
                mean[j] += this.embeddings[i][j];
            }
        }
        for (let j = 0; j < d; j++) {
            mean[j] /= n;
        }
        
        // 中心化数据
        const centered = this.embeddings.map(row => 
            row.map((val, j) => val - mean[j])
        );
        
        // 使用随机投影作为简化的降维方法
        const projectionMatrix = this.generateRandomProjection(d, 2);
        
        // 投影到2D
        const projected = centered.map(row => {
            return [
                row.reduce((sum, val, i) => sum + val * projectionMatrix[i][0], 0),
                row.reduce((sum, val, i) => sum + val * projectionMatrix[i][1], 0)
            ];
        });
        
        // 保存变换参数
        this.pcaTransform = {
            mean: mean,
            components: projectionMatrix
        };
        
        // 创建兼容的数据结构
        this.pcaData = {
            getColumn: (i) => projected.map(row => row[i])
        };
    }

    // 生成随机投影矩阵
    generateRandomProjection(inputDim, outputDim) {
        const matrix = [];
        for (let i = 0; i < inputDim; i++) {
            matrix[i] = [];
            for (let j = 0; j < outputDim; j++) {
                matrix[i][j] = (Math.random() - 0.5) * 2;
            }
        }
        
        // 归一化列向量
        for (let j = 0; j < outputDim; j++) {
            let norm = 0;
            for (let i = 0; i < inputDim; i++) {
                norm += matrix[i][j] * matrix[i][j];
            }
            norm = Math.sqrt(norm);
            for (let i = 0; i < inputDim; i++) {
                matrix[i][j] /= norm;
            }
        }
        
        return matrix;
    }

    // 可视化PCA结果
    visualizePCA() {
        if (!this.pcaData) return;

        const traces = [];
        
        // 文本分段点
        const segmentTrace = {
            x: this.pcaData.getColumn(0),
            y: this.pcaData.getColumn(1),
            mode: 'markers+text',
            type: 'scatter',
            name: '文本分段',
            text: this.segments.map(s => `分段${s.id}`),
            textposition: 'top center',
            marker: {
                size: 12,
                color: '#667eea',
                line: {
                    width: 2,
                    color: '#4c51bf'
                }
            }
        };
        traces.push(segmentTrace);

        // 如果有查询向量，也显示
        if (this.queryEmbedding && this.queryPCA) {
            const queryTrace = {
                x: [this.queryPCA[0]],
                y: [this.queryPCA[1]],
                mode: 'markers+text',
                type: 'scatter',
                name: '查询问题',
                text: ['查询'],
                textposition: 'top center',
                marker: {
                    size: 15,
                    color: '#e53e3e',
                    symbol: 'star',
                    line: {
                        width: 2,
                        color: '#c53030'
                    }
                }
            };
            traces.push(queryTrace);
        }

        const layout = {
            title: {
                text: '文本分段向量PCA降维可视化',
                font: { size: 16 }
            },
            xaxis: {
                title: '第一主成分',
                gridcolor: '#e2e8f0'
            },
            yaxis: {
                title: '第二主成分',
                gridcolor: '#e2e8f0'
            },
            plot_bgcolor: '#f7fafc',
            paper_bgcolor: '#f7fafc',
            showlegend: true,
            legend: {
                x: 1,
                y: 1
            }
        };

        Plotly.newPlot('pcaPlot', traces, layout, {responsive: true});
    }

    // 处理查询
    async processQuery() {
        const query = document.getElementById('queryInput').value.trim();
        if (!query) {
            alert('请输入查询问题');
            return;
        }

        if (this.embeddings.length === 0) {
            alert('请先生成文本分段的向量');
            return;
        }

        const button = document.getElementById('queryButton');
        button.disabled = true;
        button.textContent = '调用API中...';

        try {
            // 调用API生成查询向量
            this.queryEmbedding = await this.callEmbeddingAPI(query);
            
            // 计算查询向量的PCA投影
            this.projectQueryToPCA();
            
            // 计算相似度
            const similarities = this.calculateQuerySimilarities();
            
            // 更新可视化
            this.visualizePCA();
            
            // 显示查询结果
            this.displayQueryResults(query, similarities);
            
        } catch (error) {
            console.error('查询处理失败:', error);
            alert('处理查询时出错：' + error.message);
        } finally {
            button.disabled = false;
            button.textContent = '查询';
        }
    }

    projectQueryToPCA() {
        if (!this.queryEmbedding || !this.pcaData || !this.pcaTransform) return;

        // 中心化查询向量
        const centeredQuery = this.queryEmbedding.map((val, i) => val - this.pcaTransform.mean[i]);
        
        // 投影到2D空间
        this.queryPCA = [
            centeredQuery.reduce((sum, val, i) => sum + val * this.pcaTransform.components[i][0], 0),
            centeredQuery.reduce((sum, val, i) => sum + val * this.pcaTransform.components[i][1], 0)
        ];
    }

    calculateQuerySimilarities() {
        return this.embeddings.map((embedding, index) => {
            const similarity = this.cosineSimilarity(this.queryEmbedding, embedding);
            return {
                segmentId: index + 1,
                similarity: similarity,
                content: this.segments[index].content
            };
        }).sort((a, b) => b.similarity - a.similarity);
    }

    displayQueryResults(query, similarities) {
        const container = document.getElementById('queryResult');
        
        let html = `
            <h3>查询问题：${query}</h3>
            <h4>最相关的文本分段：</h4>
            <div class="similarity-results">
        `;
        
        similarities.slice(0, 3).forEach((result, index) => {
            html += `
                <div class="similarity-item" style="margin-bottom: 15px; padding: 15px; background: ${index === 0 ? '#f0fff4' : '#f7fafc'}; border-radius: 8px; border-left: 4px solid ${index === 0 ? '#38a169' : '#cbd5e0'};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <strong>分段 ${result.segmentId}</strong>
                        <span style="background: #667eea; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">
                            相似度: ${(result.similarity * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div style="color: #4a5568; line-height: 1.5;">
                        ${result.content.substring(0, 200)}${result.content.length > 200 ? '...' : ''}
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        container.innerHTML = html;
    }

    // 计算相似度矩阵
    calculateSimilarityMatrix() {
        if (this.embeddings.length === 0) return;

        const n = this.embeddings.length;
        this.similarityMatrix = Array(n).fill().map(() => Array(n).fill(0));

        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                this.similarityMatrix[i][j] = this.cosineSimilarity(
                    this.embeddings[i], 
                    this.embeddings[j]
                );
            }
        }

        this.displaySimilarityMatrix();
    }

    displaySimilarityMatrix() {
        if (!this.similarityMatrix) return;

        const container = document.getElementById('similarityMatrix');
        const n = this.similarityMatrix.length;
        
        let html = '<h3>分段间相似度矩阵</h3>';
        html += `<div class="similarity-matrix" style="grid-template-columns: repeat(${n + 1}, 40px);">`;
        
        // 表头
        html += '<div class="matrix-cell matrix-header"></div>';
        for (let i = 0; i < n; i++) {
            html += `<div class="matrix-cell matrix-header">${i + 1}</div>`;
        }
        
        // 矩阵内容
        for (let i = 0; i < n; i++) {
            html += `<div class="matrix-cell matrix-header">${i + 1}</div>`;
            for (let j = 0; j < n; j++) {
                const similarity = this.similarityMatrix[i][j];
                const intensity = Math.floor(similarity * 255);
                const color = `rgb(${255 - intensity}, ${255 - intensity}, 255)`;
                html += `<div class="matrix-cell" style="background-color: ${color};" title="分段${i+1} vs 分段${j+1}: ${(similarity * 100).toFixed(1)}%">
                    ${(similarity * 100).toFixed(0)}
                </div>`;
            }
        }
        
        html += '</div>';
        container.innerHTML = html;
    }

    // 余弦相似度计算
    cosineSimilarity(vecA, vecB) {
        const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
        const normA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
        const normB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
        return dotProduct / (normA * normB);
    }

    clearVisualization() {
        document.getElementById('pcaPlot').innerHTML = '';
        document.getElementById('queryResult').innerHTML = '';
        document.getElementById('similarityMatrix').innerHTML = '';
        document.getElementById('embeddingStatus').innerHTML = '';
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new RAGVisualization();
});