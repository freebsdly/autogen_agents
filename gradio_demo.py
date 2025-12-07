import sys
import os
import gradio as gr
import asyncio
import threading
import queue
from typing import Dict, Any, List, Tuple
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

try:
    # Add the parent directory to the path so we can import conf
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from conf import settings
except ImportError:
    print(
        "Failed to import settings from conf. Make sure conf is in the parent directory."
    )
    raise

deepseek_model_info = ModelInfo(
    vision=False,
    function_calling=True,  # DeepSeek支持函数调用
    json_output=True,  # DeepSeek支持JSON输出
    family="deepseek",
    structured_output=True,
)

aliyun_model_info = ModelInfo(
    vision=False,
    function_calling=True,  # DeepSeek支持函数调用
    json_output=True,  # DeepSeek支持JSON输出
    family="qwen3",
    structured_output=True,
)

volce_model_info = ModelInfo(
    vision=False,
    function_calling=True,
    json_output=True,
    family="deepseek",
    structured_output=True,
)

deepseek_model_client = OpenAIChatCompletionClient(
    api_key=settings.deepseek.api_key,
    model="deepseek-chat",
    base_url=settings.deepseek.base_url,
    model_info=deepseek_model_info,
)

aliyun_model_client = OpenAIChatCompletionClient(
    api_key=settings.aliyun.api_key,
    model="qwen3-max",
    base_url=settings.aliyun.base_url,
    model_info=aliyun_model_info,
)

doubao_model_client = OpenAIChatCompletionClient(
    api_key=settings.volces.api_key,
    model="doubao-seed-1-6-thinking-250715",
    base_url=settings.volces.base_url,
    model_info=volce_model_info,
)


# 工具定义示例
async def get_weather(location: str) -> str:
    """模拟获取天气信息"""
    return f"The weather in {location} is sunny with a temperature of 25°C."

# 创建工具实例
weather_tool = FunctionTool(get_weather, description="Get weather information for a location")

# 存储会话历史
conversation_history = []

# 定义AI助手
travel_agent = AssistantAgent(
    "TravelAgent",
    model_client=aliyun_model_client,
    tools=[weather_tool],
    system_message="You are a helpful travel agent. Help users plan their trips."
)

writer_agent = AssistantAgent(
    "WriterAgent", 
    model_client=aliyun_model_client,
    system_message="You are a helpful writer assistant. Review and improve travel plans."
)

# 定义终止条件
termination = TextMentionTermination("good job")

# 创建团队聊天
async def create_team():
    team = RoundRobinGroupChat([travel_agent, writer_agent], termination_condition=termination)
    return team

# 消息队列用于流式传输
message_queue = queue.Queue()

# 处理消息的回调函数
async def handle_message(message) -> None:
    # 处理不同类型的消息对象
    if hasattr(message, '_asdict'):
        # 如果是 NamedTuple 类型的消息
        message_dict = message._asdict()
        role = message_dict.get("source", "unknown")
        content = message_dict.get("content", "")
    elif isinstance(message, dict):
        # 如果已经是字典格式
        role = message.get("source", message.get("role", "unknown"))
        content = message.get("content", "")
    else:
        # 其他情况，尝试转换为字符串
        role = "unknown"
        content = str(message)
    
    # 将消息添加到队列中以供Gradio界面消费
    message_queue.put({"role": role, "content": content})
    # 同时保存到历史记录
    conversation_history.append({"role": role, "content": content})

# 运行团队任务
async def run_team_task(task: str):
    team = await create_team()
    # 清空历史记录
    global conversation_history
    conversation_history = []
    
    # 使用 run_stream 方法处理消息流
    async for message in team.run_stream(task=task):
        await handle_message(message)

# 包装为同步函数用于线程执行
def run_with_loop(task: str):
    # 创建新的事件循环以在线程中运行异步函数
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_team_task(task))

# 定期检查消息的函数
def update_chat_history(chat_history):
    updated_history = list(chat_history)
    
    try:
        while True:
            # 非阻塞地从队列中获取消息
            message = message_queue.get_nowait()
            
            # 解析消息并更新聊天历史
            role = message.get("role", "unknown")
            content = message.get("content", "")
            
            # 根据角色确定显示名称
            if role == "user":
                display_role = "User"
            elif role == "assistant":
                display_role = "Assistant"
            else:
                display_role = role.capitalize()
            
            # 添加到聊天历史，使用列表格式 [user_message, bot_response]
            updated_history.append([f"{display_role}", content])
    except queue.Empty:
        pass
        
    return updated_history, updated_history

# Gradio界面组件
with gr.Blocks(title="AutoGen Agents Chat") as demo:
    gr.Markdown("# AutoGen Agents Chat Interface")
    gr.Markdown("与AutoGen AI代理进行交互，查看它们之间的对话流")
    
    chatbot = gr.Chatbot(label="对话历史", height=500)
    msg = gr.Textbox(label="输入您的消息", placeholder="请输入任务描述...")
    clear = gr.Button("清除对话")
    # 添加定时器组件
    timer = gr.Timer(1, active=False)
    
    def respond(message: str, chat_history: List[List[str]]) -> Tuple[List[List[str]], gr.Timer]:
        # 在后台线程中启动任务
        thread = threading.Thread(target=run_with_loop, args=(message,))
        thread.daemon = True
        thread.start()
        
        # 返回初始状态，并激活定时器
        # 使用列表格式 [user_message, bot_response]
        return chat_history + [["User", message], ["Assistant", "正在处理中..."]], gr.Timer(active=True)
    
    def clear_conversation():
        # 清除对话历史
        global conversation_history
        conversation_history = []
        # 清空消息队列
        while not message_queue.empty():
            try:
                message_queue.get_nowait()
            except queue.Empty:
                break
        return [], gr.Timer(active=False)
    
    # 绑定事件
    msg.submit(respond, [msg, chatbot], [chatbot, timer], queue=False)
    timer.tick(update_chat_history, [chatbot], [chatbot, chatbot], queue=False)
    clear.click(clear_conversation, None, [chatbot, timer], queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)