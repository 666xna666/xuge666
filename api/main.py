import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

# 创建 FastAPI 应用实例
app = FastAPI(title="AI问答服务")

# 从环境变量读取 API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("环境变量 DASHSCOPE_API_KEY 未设置")

# 初始化阿里百炼客户端
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 定义请求体格式
class QuestionRequest(BaseModel):
    question: str

# 问答接口
@app.post("/ask")
async def ask_ai(request: QuestionRequest):
    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {"role": "system", "content": "你是一个乐于助人的助手。"},
            {"role": "user", "content": request.question}
        ],
        stream=False
    )
    answer = response.choices[0].message.content
    return {"answer": answer}

# 健康检查接口
@app.get("/health")
async def health():
    return {"status": "ok"}
