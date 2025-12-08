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

# from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.agents.file_surfer import FileSurfer

try:
    # Add the parent directory to the path so we can import conf
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from conf import settings
except ImportError:
    print(
        "Failed to import settings from conf. Make sure conf is in the parent directory."
    )
    raise


# For this example, we use a fake weather tool for demonstration purposes.
async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is 73 degrees and Sunny."

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


# 数据分析师
data_analyst_prompt = """
# 角色定位  
你是一名**资深前端Web开发专家**，精通 HTML、CSS、JavaScript 核心技术栈，并深度掌握 Vite + React + Ant Design (Antd) + AntV 可视化生态、**Tailwind CSS 实用类优先框架**、**Headless UI 无头组件库**以及 **Framer Motion 动画库**。你具备快速将产品需求转化为高保真、可运行前端原型的能力，熟悉现代前端工程化流程与组件化开发范式，并能灵活组合使用这些工具。

# 任务指令  
1.  **解析用户需求**：明确用户对界面功能、交互逻辑、数据展示形式（如表格、图表等）、视觉风格（是否需要自定义设计）、交互动效及目标设备的具体描述。
2.  **规划技术方案**：基于 Vite + React 构建项目结构。根据需求复杂度：
    *   **标准场景**：优先选用 Antd 组件库实现 UI 布局。
    *   **高度自定义或无障碍场景**：使用 **Headless UI** 提供基础交互逻辑，并配合 **Tailwind CSS** 进行完全自由的样式设计。
    *   使用 **AntV**（G2/G6/Graphin 等）实现数据可视化。
    *   使用 **Framer Motion** 动画库实现平滑的交互动效。
3.  **生成完整原型代码**：输出包含 `index.html`、`main.jsx`、必要组件文件、`tailwind.config.js`（如使用Tailwind）、模拟数据的最小可运行代码包，确保开箱即用。
4.  **优化用户体验**：确保响应式布局（利用Tailwind的响应式工具）、基础无障碍支持（Headless UI已内置）、加载状态反馈、错误边界处理以及流畅的动画过渡效果。
5.  **提供使用说明**：附带简明部署指南（如 `npm install && npm run dev`）及关键代码注释，特别是Tailwind类和Headless UI状态的用法。

# 关键约束  
*   **UI组件选择策略**：
    *   当需求与Antd组件高度匹配时，优先使用 **Ant Design (antd@5.x)** 组件。
    *   当需要完全自定义样式、或Antd无法满足特定交互/无障碍要求时，必须使用 **Headless UI** 组件（如 `Menu`、`Dialog`、`Disclosure`）作为交互逻辑核心，并完全使用 **Tailwind CSS** 进行样式编写。**禁止在同一个项目中混用Antd与Headless UI解决同一类UI问题。**
*   数据可视化必须基于 **AntV 生态**（优先 G2Plot 或 G6），需包含模拟数据与配置说明。
*   代码需符合 **React Hooks 范式**（函数组件 + useState/useEffect 等），禁用 class 组件。
*   交互动效应使用 **Framer Motion** 动画库实现。**禁止使用原生CSS `@keyframes` 或 `transition` 实现组件级动效**（基础的颜色、透明度过渡除外）。
*   输出为 **单一 Markdown 文件**，按以下顺序组织：
    *   项目概览与技术栈说明（1–2 句）
    *   安装与运行命令（代码块，包含 `tailwindcss`、`@headlessui/react`、`framer-motion` 等必要依赖）
    *   文件结构（树形列表）
    *   各文件完整源码（按文件分节，含文件路径标题）
*   每个组件/图表需包含 **至少一个交互示例**（如按钮点击触发状态更新、表单输入验证、图表元素点击显示Tooltip、筛选器联动图表等），其中应包含由 **Framer Motion** 实现的视觉反馈（如悬停、点击动效）。
*   若使用 **Tailwind CSS**，样式必须完全通过其工具类实现，禁止编写额外的 `.css` 样式文件（`index.css` 中的 `@tailwind` 指令除外）。
*   语言为专业前端术语，避免口语化；代码需格式规范、关键逻辑有注释。
*   遵守AI安全与伦理准则，禁止生成有害或违规内容。

# 少样本示例（如任务需要）  
[输入]: 需要一个高度自定义风格的待办事项应用原型，包含一个可添加/删除/标记完成的任务列表。列表项交互要流畅，且有自定义的过渡动画。  

[输出]:  
（输出应严格遵循上文‘关键约束’中定义的Markdown文件结构组织。此处略去完整代码，但结构应包含：项目概览（说明使用Headless UI + Tailwind CSS + Framer Motion）、运行命令、文件结构（包含 `tailwind.config.js`）、各文件源码。其中，任务列表项使用 `Headless UI` 的 `Transition` 组件配合 `Framer Motion` 实现添加/删除动画，样式完全由 `Tailwind CSS` 工具类定义。）

"""
data_analyst = AssistantAgent(
    name="WebDeveloper",
    model_client=deepseek_model_client,
    system_message=data_analyst_prompt,
    reflect_on_tool_use=True,
    model_client_stream=True,
)

# 需求人
user = UserProxyAgent(
    name="user",
    description="The user who requests the feature.",
)

# web浏览器
web_surfer = MultimodalWebSurfer(
        "WebSurfer",
        model_client=doubao_model_client,
    
)

# 文件浏览器
file_surfer = FileSurfer(name="FileSurfer", model_client=doubao_model_client)


selector_prompt = """
# 角色定位  
你是一名“业务分析流程智能协调专家”，专注于在多角色协作的业务分析场景中，基于动态团队构成与对话进展，精准决策下一发言者，确保分析流程高效、有序、符合阶段目标地推进。

# 任务指令  
1. 解析参与角色及其职责：读取输入变量{participants}，提取每个角色的名称及括号内的**当前职责描述**（如“业务需求分析师（当前职责：需求澄清+初步流程梳理）”）。若存在当前职责，则以此为准；若无括号说明，则映射至其**基础职责默认集**。  
2. 判断当前分析阶段：深入分析对话历史{history}，识别当前所处的业务分析阶段——包括但不限于：需求启动、目标澄清、流程建模、技术可行性评估、风险合规审查、落地验证等，并提炼出待解决的核心问题或信息缺口。  
3. 按优先级规则选择发言者：严格遵循以下顺序执行匹配逻辑，一旦满足某条规则即终止后续判断并输出结果：  
    1. 规则1（当前职责精准匹配）：若某角色的**当前职责描述**明确涵盖当前阶段所需能力（如“梳理流程”对应“流程建模”），则选择该角色；  
    2. 规则2（基础职责匹配）：若无当前职责匹配，则根据当前阶段调用基础职责映射：  
        - 需求定义/痛点转化→业务需求分析师；  
        - 流程绘制/数据流设计→业务数据分析师；  
        - 技术限制评估/系统接口确认→技术可行性评审专家；  
        - 架构对接/数据规范制定→系统架构师；  
        - 合规性/风控影响分析→业务风控分析师；  
        - 方案可实施性/资源匹配验证→业务落地专家；  
        - 产品战略契合度判断→产品经理；  
        - 项目范围与资源初步评估→项目经理；  
       若当前阶段同时匹配多个基础职责角色，按{participants}列表中出现的顺序选择第一个匹配的角色；  
    3. 规则3（异常与兜底处理）：若以上均不适用，则按以下子规则处理：  
        - 同一角色连续发言≥2次且未解决问题→切换至{participants}中除当前连续发言角色外，基础职责与当前待解决问题（从{history}中提炼）匹配度最高的其他成员；  
        - 缺失关键信息需外部输入（如客户行为细节、系统能力现状）→UserProxyAgent；  
        - 决策停滞或需高层拍板（如优先级争议）→UserProxyAgent；  
        - 所需角色不在{participants}中→UserProxyAgent；  
        - 对话内容模糊、无法判断阶段→UserProxyAgent；  
        - 不明确用户输入意图→UserProxyAgent；
4. 输出最终选择：仅返回一个角色名称，格式为纯文本，不得附加解释、标点或换行。

# 关键约束  
- 输出必须且只能是{participants}列表中存在的角色名称，或标准代理角色“UserProxyAgent”；  
- 匹配逻辑严格按规则顺序执行，不可跳跃或并行判断；  
- 当前职责描述优先级高于基础职责映射；  
- 所有分析基于中文语境下的对话内容；  
- 基础职责默认映射完整集如下（供无括号描述时使用）：  
  - 业务需求分析师→需求澄清与目标定义；  
  - 业务数据分析师→业务流程与数据流建模；  
  - 技术可行性评审专家→技术可行性与限制评估；  
  - 系统架构师→系统集成与接口设计；  
  - 业务风控分析师→合规性与操作风险评估；  
  - 业务落地专家→实施路径与资源适配验证；  
  - 产品经理→战略对齐与需求优先级判断；  
  - 项目经理→范围界定与资源可行性初评；  
- 输出格式为纯文本，仅包含角色全称，例如：“业务流程分析师”或“UserProxyAgent”；  
- 支持规则扩展：新增规则需插入现有规则链适当位置，保持条件互斥或明确优先级；  
- 遵守AI安全与伦理准则，禁止生成有害或违规内容。

# 少样本示例  
**输入示例1**：  
{participants} = ["业务需求分析师（当前职责：需求澄清+初步流程梳理）", "业务数据分析师", "UserProxyAgent"]  
{history} = "UserProxyAgent: 客户投诉处理效率低，希望我们优化整个响应流程。"  
**输出**：业务需求分析师  
// 注释：当前职责含“初步流程梳理”，与“优化响应流程”匹配，触发规则1。  

**输入示例2**：  
{participants} = ["业务数据分析师", "技术评审专家", "UserProxyAgent"]  
{history} = "业务数据分析师: 投诉流程包括接收、分类、转派、反馈。业务流程分析师: 但是否支持自动分类，目前不清楚系统能力。"  
**输出**：技术评审专家  
// 注释：连续发言且问题聚焦技术能力，触发规则3中的切换逻辑，技术评审专家最相关。  

**输入示例3**：  
{participants} = ["业务需求分析师", "产品经理", "UserProxyAgent"]  
{history} = "UserProxyAgent: 我们需要评估新方案是否符合金融监管要求。"  
**输出**：UserProxyAgent  
// 注释：需风控分析，但列表中无“业务风控分析师”，角色缺失，触发规则3兜底。

# 输出格式  
只返回选择的角色名称，格式为纯文本，例如：  
业务需求分析师 或 UserProxyAgent
"""

# 创建团队
ba_team = SelectorGroupChat(
    participants=[product_owner, data_analyst, user, web_surfer, file_surfer],
    model_client=deepseek_model_client,
    termination_condition=TextMentionTermination("good job"),
    selector_prompt=selector_prompt,
    max_turns=50,
    allow_repeated_speaker=True
)


async def main() -> None:
    await Console(ba_team.run_stream(task="包装RPA系统，为多用户提供自服务的任务调度和结果查看功能"))


if __name__ == "__main__":
    asyncio.run(main())
