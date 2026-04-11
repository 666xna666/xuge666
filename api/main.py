import os
from fastapi import FastAPI, Request
from openai import OpenAI
import uvicorn

app = FastAPI()

# 初始化阿里百炼客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

@app.post("/process")
async def process(request: Request):
    # 1. 解析百炼平台传入的 JSON 数据
    data = await request.json()
    
    # 提取用户输入的最后一条文本消息
    user_messages = data.get("input", [])
    last_message = ""
    if user_messages:
        # 取最后一个 role 为 user 的消息内容
        for msg in reversed(user_messages):
            if msg.get("role") == "user":
                content = msg.get("content", [])
                if content and content[0].get("type") == "text":
                    last_message = content[0].get("text", "")
                    break

    if not last_message:
        return {"error": "未收到有效提问"}

    # 2. 调用大模型 API
    try:
        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {"role": "system", "content": "你是一个乐于助人的助手。"},
                {"role": "user", "content": last_message}
            ],
            stream=False
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"调用 AI 服务出错：{str(e)}"

    # 3. 按照百炼要求的格式返回
    return {
        "output": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": answer
                    }
                ]
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
