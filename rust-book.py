from unstructured.partition.md import partition_md
from unstructured.chunking.title import chunk_by_title
from sentence_transformers import SentenceTransformer, util
import numpy as np
import nltk

# --------------------------
# 初始化工具（强制CPU运行）
# --------------------------
nltk.download('punkt')
# 轻量化模型（80MB，CPU推理极快）
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# --------------------------
# 1. 解析Markdown文档（核心：保留代码块完整性）
# --------------------------
# 本地Markdown文件（示例：含Python/Rust代码块的技术文档）
md_file_path = "D:\\markdowns\\book\\src\\ch03-02-data-types.md"
elements = partition_md(
    md_file_path,
    # 关键配置：识别并保留代码块不拆分
    extract_code_block_fences=True,  # 识别```python/```rust等代码块
    extract_images=False,  # 关闭图片提取（减少CPU消耗）
    skip_infer_table_types=[],  # 关闭表格推理（无表格时）
    # 适配技术文档的Markdown分隔符
    separators=["\n\n", "\n", "### ", "## ", "# ", ". "],
)

# --------------------------
# 2. 按Markdown结构粗切分（标题/代码块优先）
# --------------------------
# 按标题层级切分（保证同主题内容在一个chunk）
structured_chunks = chunk_by_title(
    elements,
    max_characters=600,  # 单chunk字符数（可根据代码块长度调整）
    overlap=50,  # 重叠字符保证文本语义连续
    multipage_sections=True,
)

# --------------------------
# 3. 语义精切分（仅处理文本，代码块直接保留）
# --------------------------
final_chunks = []
for chunk in structured_chunks:
    chunk_text = chunk.text.strip()
    
    # 规则1：代码块直接保留为独立chunk（不拆分）
    if "```" in chunk_text:
        final_chunks.append(chunk_text)
        continue
    
    # 规则2：纯文本按语义切分（避免跨语义拆分）
    sentences = nltk.sent_tokenize(chunk_text)
    if len(sentences) <= 1:
        final_chunks.append(chunk_text)
        continue
    
    # 计算句子语义嵌入（CPU优化：禁用Tensor，用Numpy）
    embeddings = model.encode(
        sentences,
        convert_to_tensor=False,
        batch_size=8,  # CPU最优批量大小（8-16）
        show_progress_bar=False,
    )
    
    # 计算相邻句子的余弦相似度（技术文档阈值：0.45）
    similarities = []
    for i in range(len(embeddings)-1):
        sim = np.dot(embeddings[i], embeddings[i+1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
        similarities.append(sim)
    
    # 确定语义切分点（低于阈值则切分）
    split_points = [i+1 for i, sim in enumerate(similarities) if sim < 0.45]
    start = 0
    for point in split_points:
        final_chunk = " ".join(sentences[start:point]).strip()
        if final_chunk:  # 过滤空chunk
            final_chunks.append(final_chunk)
        start = point
    # 处理最后一段
    final_chunk = " ".join(sentences[start:]).strip()
    if final_chunk:
        final_chunks.append(final_chunk)

# --------------------------
# 输出结果（代码块完整，文本按语义切分）
# --------------------------
for i, chunk in enumerate(final_chunks):
    print(f"=== Chunk {i+1} ===")
    print(chunk)
    print("-" * 100)
