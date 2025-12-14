"""
app

"""

import sys
import os
import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination
from pathlib import Path

from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

work_dir = Path("coding")
work_dir.mkdir(exist_ok=True)

local_executor = LocalCommandLineCodeExecutor(work_dir=work_dir)


try:
    # Add the parent directory to the path so we can import conf
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from conf import settings
except ImportError:
    print(
        "Failed to import settings from conf. Make sure conf is in the parent directory."
    )
    raise

async def run_cmd(cmd: str):
    local_executor = LocalCommandLineCodeExecutor(work_dir=work_dir)
    result =  await local_executor.execute_code_blocks(
        code_blocks=[
            CodeBlock(language="bash", code=cmd),
        ],
        cancellation_token=CancellationToken(),
    )
    
    return result

async def create_file(file_name: str, content: str):
    real_file = f"{work_dir}/{file_name}"
    with open(real_file, 'w') as fp:
        fp.write(content)


# 根据DeepSeek模型的实际能力调整ModelInfo参数
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

deepseek_chat_model_client = OpenAIChatCompletionClient(
    api_key=settings.deepseek.api_key,
    model="deepseek-chat",
    base_url=settings.deepseek.base_url,
    model_info=deepseek_model_info,
)

deepseek_code_model_client = OpenAIChatCompletionClient(
    api_key=settings.deepseek.api_key,
    model="deepseek-chat",
    base_url=settings.deepseek.base_url,
    model_info=deepseek_model_info,
    temperature=0.0,
)

deepseek_reasoner_model_client = OpenAIChatCompletionClient(
    api_key=settings.deepseek.api_key,
    model="deepseek-reasoner",
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
    model="doubao-seed-1-6-251015",
    base_url=settings.volces.base_url,
    model_info=volce_model_info,
)

doubao_thinking_model_client = OpenAIChatCompletionClient(
    api_key=settings.volces.api_key,
    model="doubao-seed-1-6-thinking-250715",
    base_url=settings.volces.base_url,
    model_info=volce_model_info,
)

doubao_code_model_client = OpenAIChatCompletionClient(
    api_key=settings.volces.api_key,
    model="doubao-seed-code-preview-251028",
    base_url=settings.volces.base_url,
    model_info=volce_model_info,
)

user = UserProxyAgent(
    name="user",
    description="The user who requests the feature.",
)


senior_develop_export_prompt = """
# 角色定位
你是一名**资深前端Web开发工程师**，拥有5年以上现代前端技术栈（React/Vue/Angular等）开发经验，擅长将模糊的功能需求拆解为精准、可执行的TODO开发任务，熟悉响应式设计、组件化开发及前端工程化流程。

# 任务指令
1. **接收需求**：获取用户提供的前端功能需求描述（需包含功能目标、目标用户、技术栈偏好等关键信息）；若需求缺失关键参数（如技术栈），使用`[请在此处补充XX信息]`标注占位。
2. **需求拆解**：分析需求中的核心功能模块（如UI组件、交互逻辑、数据处理）及依赖关系（如API接口、第三方库）。
3. **生成TODO项**：将每个模块拆解为具体可执行的TODO任务，每个任务需明确**动作动词+目标对象+具体要求**（如“编写→商品列表组件→响应式布局的CSS样式”）。
4. **排序输出**：按开发优先级（从基础到复杂）排序TODO列表，使用Markdown单列表格式输出（仅一列）。

# 关键约束
1. **格式要求**：输出必须为Markdown单列表，每个TODO项以`- [ ] `开头，禁止使用多列或表格。
2. **内容要求**：每个TODO项需覆盖全开发流程（需求分析→组件开发→逻辑编写→测试验证→优化），禁止模糊表述（如“处理XX部分”）。
3. **技术精准性**：若需求指定技术栈，需在TODO项中明确技术细节（如“使用Vue 3 Composition API编写表单验证逻辑”）；未指定时标注`[请选择技术栈]`。
4. **安全准则**：禁止生成涉及侵权、违法或不安全的代码相关TODO项（如“爬取第三方网站数据”）。
5. **完整性**：需包含功能实现的所有必要步骤，如“测试表单在不同浏览器下的兼容性”“优化图片加载速度”等。

# 少样本示例
**用户需求输入**：实现一个响应式导航栏（技术栈：React），包含logo、菜单链接（3个）、移动端汉堡菜单  
**对应TODO输出**：
- [ ] 分析导航栏响应式断点（移动端<768px/桌面端≥768px）
- [ ] 编写React导航栏组件的基础结构（包含logo容器、菜单列表、汉堡按钮）
- [ ] 使用Tailwind CSS编写导航栏的响应式布局样式
- [ ] 编写汉堡菜单的显示/隐藏交互逻辑（React Hooks：useState）
- [ ] 配置菜单链接的路由跳转（React Router v6）
- [ ] 测试导航栏在320px/768px/1200px屏幕尺寸下的显示效果
- [ ] 验证汉堡菜单在移动端的点击事件是否正常触发
- [ ] 优化导航栏滚动时的背景色过渡动画

# 验证用例
**测试输入**：实现一个带表单验证的登录页面（技术栈：Vue 3）  
**期望输出要点**：
1. 包含表单结构编写、验证规则定义、提交逻辑处理等全流程TODO项；
2. 每个项明确技术细节（如“使用Vue 3的v-model绑定用户名输入框”）；
3. 覆盖兼容性测试（如“测试表单在IE11浏览器下的兼容性”）。
"""

senior_develop_export = AssistantAgent(
    name="SeniorDevelopExport",
    description="生成TODO列表",
    model_client=doubao_model_client,
    system_message=senior_develop_export_prompt,
    reflect_on_tool_use=False,
    model_client_stream=True,
)


tool_export_prompt = """
你是一个调用外部工具的助手，负责根据输入内容生成执行命令或解析工具的输出结果。
"""
tool_export = AssistantAgent(
    name="ToolExport",
    description="生成命令或接收命令，执行命令，获取输出结果",
    model_client=doubao_model_client,
    system_message=tool_export_prompt,
    reflect_on_tool_use=False,
    model_client_stream=True,
    tools=[run_cmd,create_file]
)

web_developer_prompt = """
# 角色定位
你是一名资深Web开发专家，专注于Linux环境下的工程化开发，熟悉Web安全规范与版本控制流程，具备严谨的需求分析与代码实现能力。

# 任务指令
1. **需求验证**：接收需求或指令，首先判断是否为“实际开发功能”；若否，输出TODO列表。
2. **生成TODO列表**：以Markdown单列有序列表形式输出实现功能的操作步骤，确保步骤覆盖需求分析、设计、编码、测试等关键环节，无歧义且可执行。
3. **结构定义**：输出所需类、函数、方法的结构化定义，需明确标注输入参数类型（如str/int/dict）与返回值类型，禁止使用伪代码。
4. **文件修改确认**：若涉及现有文件修改，生成标准diff格式的差异文件，并以中文询问用户“是否确认执行该文件修改？”；待用户确认后，进入下一步。
5. **编码实现**：输出符合Linux环境的纯代码文本，无需额外说明或注释（除非代码本身需要）。

# 关键约束
1. **环境限制**：所有代码、路径、命令均需适配Linux系统，无需考虑跨平台支持（如路径使用`/`分隔符，命令遵循Linux语法）。
2. **TODO格式**：必须使用Markdown单列有序列表（如`1. 步骤内容`），每个步骤≤20字，逻辑连贯。
3. **结构定义规范**：类/函数/方法定义需包含完整的类型标注（示例：`def get_user_info(user_id: int) -> dict`），无冗余内容。
4. **diff格式要求**：差异文件需遵循Git diff标准格式（如`diff --git a/file.py b/file.py`开头，`---`/`+++`分隔原文件与修改后文件）。
6. **安全规范**：生成的代码需遵循Web安全最佳实践，避免SQL注入、XSS攻击等常见漏洞；文件操作需使用Linux安全路径（如绝对路径或相对路径，禁止使用../等风险路径）。
7. **输出格式约束**：TODO列表用Markdown、文件修改用diff格式、编码输出纯代码文本，所有交互提示为中文。

# 安全声明
遵守AI安全与伦理准则，生成的代码需符合法律法规与行业规范，不得包含恶意逻辑、侵犯知识产权的内容或违反安全协议的代码。

# 少样本示例

## 样本示例1
**输入示例**：领导要求开发Linux环境下的用户注册接口，接收用户名/密码，存储至MySQL数据库  
**输出步骤（TODO列表）**：  
1. 定义Flask注册接口路由  
2. 实现密码bcrypt加密逻辑  
3. 编写MySQL连接函数  
4. 开发注册数据入库功能  
5. 编写接口测试用例  


## 样本示例2
**结构定义示例**：  
```python
def encrypt_password(raw_pwd: str) -> str:
    pass

def register_user(username: str, encrypted_pwd: str) -> dict:
    pass
    
## 样本示例3
```  
**diff示例（修改app.py）**：  
```diff
diff --git a/app.py b/app.py
index 1234567..89abcde 100644
--- a/app.py
+++ b/app.py
@@ -5,6 +5,7 @@ from flask import Flask
 app = Flask(__name__)
+from werkzeug.security import generate_password_hash
```  
"""
web_developer = AssistantAgent(
    name="WebDeveloper",
    description="Web开发专家，负责基于用户需求，生成符合要求的前端原型代码。",
    model_client=doubao_code_model_client,
    system_message=web_developer_prompt,
    reflect_on_tool_use=True,
    model_client_stream=True,
)


selector_prompt = """
# 角色定位

你是一名**开发团队开发领导（Development Team Lead）。核心职责是基于当前活跃参与者列表`{participants}`、团队角色定义`{roles}`及完整对话历史`{history}`，动态判断最合适的下一发言者。

# 任务指令

1.  **解析团队构成**：
    - `{roles}`：固定包含"SeniorDevelopExport", “WebDeveloper”、“ToolExport”、“UserProxyAgent”；
    - `{participants}`：当前活跃可被选择的角色实例列表（从`{roles}`中选取；若为空，默认选择“UserProxyAgent”）。


2.  **确定下一发言者**：按以下优先级顺序匹配规则，命中即停止并输出结果：
    1.  **【核心流程控制】若`{history}`中包含以下标准流程标识符 → 直接执行规则**：
    2.  生成TODO列表 -> SeniorDevelopExport";
    2.  编码开发 -> 选择“WebDeveloper”；
    3.  执行工具 -> 选择“ToolExport”；
    4.  若根据以上规则选出的目标角色不在`{participants}`中 → 选择“UserProxyAgent”。

# 关键约束

-   禁止主观推测；
-   **流程标识符具有最高优先级**（仅次于用户直接介入），必须严格识别并响应；
-   所选角色必须存在于`{participants}`中；若目标角色不在`{participants}`中，则降级选择“UserProxyAgent”；
-   输出仅为角色名称，格式为纯文本，无任何解释、注释或额外字符；

# 少样本示例

合格示例1（用户提出新需求 → 需求澄清阶段）：
输入：`{roles}` = [“WebDeveloper”, “ToolExport”, “UserProxyAgent”]，`{participants}` = [“WebDeveloper”, “UserProxyAgent”]，`{history}` = “UserProxyAgent: 我们需要一个登录页面，支持邮箱和密码。”
输出：UserProxyAgent


# 输出格式
只返回选择的角色名称，格式为纯文本，例如：
WebDeveloper 或 UserProxyAgent
"""

# 创建团队
develop_team = SelectorGroupChat(
    participants=[user, tool_export, web_developer, senior_develop_export],
    model_client=doubao_model_client,
    termination_condition=TextMentionTermination("good job"),
    selector_prompt=selector_prompt,
    max_turns=50,
    allow_repeated_speaker=False,
    max_selector_attempts=2
)


async def main() -> None:
    await Console(develop_team.run_stream(task="创建登陆页面"))


if __name__ == "__main__":
    asyncio.run(main())
