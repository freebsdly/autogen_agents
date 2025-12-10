from sentence_transformers import SentenceTransformer
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

model = SentenceTransformer('all-MiniLM-L6-v2')

# 1. 加载Embedding模型（生成向量）
# model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

# 2. 待存入的原始文本数据
texts = [
    "如何本地部署Embedding模型？",
    "开源中文Embedding模型有哪些？",
    "BGE模型适合中文检索场景吗？"
]

# 3. 文本向量化（核心步骤：转换为数值向量）
embeddings = model.encode(texts, normalize_embeddings=True)  # 输出(3, 512)的向量数组

# 4. 创建Qdrant集合（指定向量维度和距离算法）
collection_name = "chinese_embedding_demo"
# client.create_collection(
#     collection_name=collection_name,
#     vectors_config=VectorParams(
#         size=512,  # 匹配模型的向量维度（bge-small-zh是512维）
#         distance=Distance.COSINE  # 余弦相似度（Embedding检索首选）
#     )
# )

# 5. 向量存入Qdrant（可附带原始文本作为元数据）
points = [
    PointStruct(
        id=i,  # 唯一ID
        vector=embeddings[i].tolist(),  # 向量数据（需转列表）
        payload={"text": texts[i]}  # 附带原始文本，方便检索后展示
    )
    for i in range(len(texts))
]

# 批量插入向量
client.upsert(collection_name=collection_name, points=points)
print("向量已成功存入Qdrant！")
