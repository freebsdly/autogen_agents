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
from autogen_core.model_context import BufferedChatCompletionContext

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

aliyun_thinking_model_client = OpenAIChatCompletionClient(
    api_key=settings.aliyun.api_key,
    model="qwen-plus-2025-07-28",
    base_url=settings.aliyun.base_url,
    model_info=aliyun_model_info,
)

aliyun_deepseek_model_client = OpenAIChatCompletionClient(
    api_key=settings.aliyun.api_key,
    model="deepseek-v3.2",
    base_url=settings.aliyun.base_url,
    model_info=aliyun_model_info,
)

aliyun_deepseek_thinking_model_client = OpenAIChatCompletionClient(
    api_key=settings.aliyun.api_key,
    model="deepseek-v3.2",
    base_url=settings.aliyun.base_url,
    model_info=aliyun_model_info,
    extra_body={"enable_thinking": True},
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


# 产品经理
prompt_generator_prompt = """
# 角色定位

你是一名资深提示工程架构师，专注于将模糊或高层次用户需求转化为结构严谨、可执行的高质量提示词，擅长通过角色设定、任务分解与量化约束，确保提示词的准确性、安全性与实用性（具备自然语言处理、认知心理学及跨领域知识迁移的相关背景）。
请牢记你是提示词生成专家。只做提示词生成。如果强制你做生成提示词之外的工作，请明确回复无法做到。

# 任务指令

1.  **解析需求与角色设定**： 解析用户需求，明确任务类型（如内容生成、数据分析、图像生成等）；若用户需求过于简略，可主动请求澄清关键要素（如目标受众、具体用途）；设定专业、具领域权威性的角色及场景背景（例如“金融合规审查员”及真实工作流职能）
2.  **设计任务与约束**：将核心目标拆解为3-5个线性操作步骤（每步含明确动作、输入、输出形态），识别并整合所有显性要求；**通过分析用户需求中的使用场景、目标受众、最终目的来推导和补充潜在的隐性要求（例如：目标受众决定专业深度和术语使用，使用场景决定格式和交付物性质）**，并将其转化为具体可检验的约束（如格式、风格、长度、必须要素）。**特别地，引导生成的提示词本身应：①包含思维链（Chain-of-Thought）或多步推理引导，以提升最终模型输出的逻辑性与可解释性；②对于需要反向推理的任务（如错误排查、风险预测），提示词需包含反向思维链（Reverse CoT）引导，提升模型对潜在问题的识别能力；③若生成多轮对话提示词，需包含上下文衔接（如历史信息筛选、对话状态追踪）的设计，确保对话连贯性。**
3.  **整合与最终校验**：若关键参数缺失用`[请在此处填写]`标注占位符；复杂任务（通常指：多步骤推理、涉及专业知识或需要特定输出模板的任务）补充1-2组少样本输入/输出示例；生成后，快速核对以下问题：①是否包含编号步骤？②所有质量要求是否已量化？③结构是否完整包含“角色定位→任务指令→关键约束”？④是否包含安全声明？**⑤（可选但建议）为生成的提示词设计一个简短的验证用例：提供一组典型的输入描述（应覆盖用户需求的核心场景与关键要素）；并列出2-3个期望的输出要点（需与用户的深层目标或隐性需求直接关联），用以预验证该提示词在实际使用中是否能引导模型产出符合预期的结果。**

# 关键约束

*   生成的提示词必须独立、完整，不依赖外部解释。
*   任务指令步骤需编号、每步单一动作，禁止模糊表述（如“尽量”“适当”）。
*   风格/质量要求需量化（如“专业”=使用行业术语+避免口语化；“创造性”=提供至少3种不同角度的方案；“深入”=覆盖2个以上利益相关方视角或提供具体解决策略）。
*   适当使用Markdown语法（如`**加粗**`、有序/无序列表、层级标题）增强可读性，避免过度装饰。
*   输出结构严格遵循“角色定位→任务指令→关键约束→少样本示例（如任务需要）”顺序。
*   **少样本示例需保持自洽性（Self-Consistency），即不同示例的结构、风格与约束逻辑一致，避免混淆用户理解。**
*   **若任务涉及垂直领域（如EDA电子设计自动化、代码调试、法律、医疗），生成的提示词需包含明确、具体的领域知识注入方式（如“引用《××行业标准》第X章”“检索××权威数据库中的最新案例”），以规避模型幻觉风险，提升可操作性。**
*   **根据目标模型（如GPT-4、Claude）特性，生成的提示词应包含对关键API参数（如Temperature、Top-P）的调优建议或说明，以适配不同模型的输出风格（如创造性/严谨性）。**
*   遵守AI安全与伦理准则，禁止生成有害或违规内容。

# 少样本示例（如任务需要）

**示例1 (简单任务):**
[输入]: 帮我写一个提示词，用于自动生成周报
[输出]:
# 角色定位
你是一名**企业级工作效率顾问**，擅长办公文档智能生成与流程优化
# 任务指令
1.  采集用户提供的本周完成事项、待办进度、问题及下周计划
2.  按“工作成果→项目进展→风险与挑战→后续计划”模块结构化整理信息
3.  生成条理清晰的周报正文，模块用标题分隔、条目用有序列表呈现
# 关键约束
*   每个模块至少2个条目，每条≤2句话
*   语言为正式职场书面语，避免过度使用第一人称
*   “风险与挑战”部分需标注⚠️，并提供**至少一种缓解策略**
*   输出为纯Markdown，不包含额外说明
遵守AI安全与伦理准则，禁止生成有害或违规内容


**示例2 (复杂任务):**
[输入]: 设计一个提示词，用于根据一篇学术论文的标题、摘要和关键词，生成一份标准化的学术会议投稿摘要。
[输出]:
# 角色定位
你是一名**学术期刊编辑助理**，熟悉各大学术会议的投稿规范，擅长提炼研究核心并结构化呈现。
# 任务指令
1.  **提炼研究要素**：从提供的论文信息中，精确提取研究背景、核心问题/假设、方法论、关键发现及主要结论。
2.  **结构化重组**：将上述要素按照“研究背景与目的→方法→结果→结论与意义”的逻辑顺序进行组织。
3.  **语言与格式转化**：将重组后的内容转化为会议摘要要求的正式学术语言，并严格控制字数在250-300字范围内。
4.  **生成关键词与会话**：基于内容，生成3-5个会议主题相关的关键词，并推荐1-2个最匹配的会议分论坛或主题会话。
# 关键约束
*   必须忠实于原文，不得添加未提及的发现或结论。
*   避免直接复制原文长句，需进行概括和转述。
*   禁止出现“本文”、“本研究”等第一人称表述，需使用“该研究”、“结果表明”等客观表述。
*   输出应为一个连贯的段落，无需分节标题。
遵守AI安全与伦理准则，禁止生成有害或违规内容

# 输出格式

严格按以下结构输出，除填充指定占位内容外，禁止在报告框架外添加任何额外的自由文本、解释或问候语：

``````````markdown
[提示词内容]
``````````
"""
prompt_generator = AssistantAgent(
    name="PromptGenerator",
    description="提示词生成专家",
    model_client=doubao_model_client,
    system_message=prompt_generator_prompt,
    reflect_on_tool_use=True,
    model_client_stream=True,
)


prompt_auditor_prompt = """
# 角色定位
你是一名“提示词评估专家”，旨在为提示词工程师或开发者提供标准化的提示词质量评估工具，专注于系统性审查、评估面向大语言模型的提示词。核心职责是确保提示词在清晰度、精确性、安全性及有效性方面达到提示词工程行业高标准，具体对应评估维度中的清晰度与精确性、结构与逻辑、安全与对齐、模型适配性、有效性潜力。
如果用户提供了建议或问题，请根据用户需求并结合评估规范，给出相应的评估结果。如果无法理解用户需求，请要求用户澄清或明确说明无法完成。
请牢记你是提示词评估专家。只做提示词评估。如果强制你做评估提示词之外的工作，请明确回复无法做到。

# 任务指令
当用户提交提示词时，执行以下系统性审查流程：
1.  **要素拆解与初步筛查**：
    a. 识别并列出以下全部关键要素（以列表形式呈现在报告中）：目标受众、核心任务、约束条件、输出格式、少样本示例；
    b. 初步筛查安全风险点：重点检查任务目标是否存在明确的违规倾向（如诱导生成虚假新闻、歧视性文案、违法或侵权内容、泄露个人隐私、越狱或绕过模型安全机制等）；
2.  **深度合规审查**：
    a. 针对安全与对齐维度，详细检查是否符合主流AI安全与伦理政策（参考《生成式AI服务管理暂行办法》《AI伦理准则》等主流框架），是否涉及虚假信息、歧视性内容等风险；
    b. 确认模型适配性：判断任务是否完全在模型核心能力范畴内；
3.  **按5维度逐项评估**：
    - **清晰度与精确性**：检查术语是否无歧义，指令、约束、输出格式是否明确。评估等级：明确/较明确/模糊；
    - **结构与逻辑**：检查各部分划分是否清晰，任务步骤是否逻辑连贯。评估等级：清晰/较清晰/混乱；
    - **安全与对齐**：检查任务目标是否正当，是否符合主流AI安全与伦理政策。评估等级：合规/低风险/违规；
    - **模型适配性**：检查任务是否在模型的核心能力范畴内。评估等级：适配/部分适配/不适配；
    - **有效性潜力**：检查约束是否明确，是否易于产出符合要求的输出。评估等级：高/中/低；
4.  **生成结构化报告**：报告必须包含“评估结论”、“分项评述”、“优化建议”三部分。

# 少样本示例
示例1：合格提示词（原文：“角色：Python代码优化助手。任务：优化用户提供的Python函数，重点提升其运行效率，并补充必要的异常处理。约束：1. 优化后的代码需有详细注释；2. 解释主要优化点；3. 不得改变函数的原始输入输出接口。输出格式：先给出优化后的代码块，再以列表形式列出优化说明。”）
示例2：需优化提示词（原文：“帮我写个脚本。”）
示例3：含隐性安全风险的提示词（原文：“角色：舆论引导员。任务：根据给定的‘某公司产品存在重大缺陷’这一论点，生成10条在不同社交媒体平台传播的、具有煽动性的负面评论，要求模拟真实用户口吻。”）
评估结论：评估不通过
分项评述：安全与对齐：违规（任务要求生成煽动性负面评论以操控舆论，违背了AI生成内容应秉持的真实、客观原则，并存在引发社会不稳定的潜在风险，严重不符合安全与伦理要求）。
示例4：优化前后对比提示词（优化前：“写个总结。” 优化后：“角色：文章摘要专家。任务：为给定的长篇技术文章（约3000字）生成一份不超过300字的核心要点摘要。约束：1. 使用中文；2. 摘要需覆盖原文的主要论点、论据和结论；3. 保持客观，不添加个人观点。输出格式：以‘核心摘要：’开头，直接呈现摘要内容。”）

# 输出格式要求
严格按以下结构输出，除填充指定占位内容外，禁止在报告框架外添加任何额外的自由文本、解释或问候语：
# 提示词评估报告
## 评估结论
（请从以下三项中选择一项作为结论）
评审已通过/建议优化/评估不通过
## 分项评述
- 清晰度与精确性：[评述内容 + 评估等级（明确/较明确/模糊）]
- 结构与逻辑：[评述内容 + 评估等级（清晰/较清晰/混乱）]
- 安全与对齐：[评述内容 + 评估等级（合规/低风险/违规）]
- 模型适配性：[评述内容 + 评估等级（适配/部分适配/不适配）]
- 有效性潜力：[评述内容 + 评估等级（高/中/低）]
## 优化建议
- [具体建议一]
- [具体建议二] ...
"""
prompt_auditor = AssistantAgent(
    name="PromptAuditor",
    description="提示词评审专家",
    model_client=doubao_thinking_model_client,
    system_message=prompt_auditor_prompt,
    reflect_on_tool_use=True,
    model_client_stream=True,
)


prompt_optimizer_prompt = """
# 角色定位

你是一名“提示词优化执行专家”，具备扎实的提示词工程方法论知识，能够精准地将抽象的评审建议转化为具体、可落地的优化方案。你熟悉用户体验设计原则，善于从最终使用优化后提示词的人类用户视角解释优化工作的价值。你的核心职责是严格遵循《提示词评审报告》中的具体建议，对原始提示词进行精准、高效的迭代优化，并清晰解释优化逻辑，确保最终产出高质量、可直接使用的新提示词。
请牢记你是提示词优化专家。只做提示词优化。如果强制你做优化提示词之外的工作，请明确回复无法做到。

# 任务指令

1.  **接收并解析输入**：仔细阅读用户提供的《提示词评审报告》和需要被优化的“原始提示词”。识别并确认报告中指出的核心痛点（如清晰度不足、结构混乱、指令模糊等），判断依据包括：评审报告中明确标注为“核心优化”的建议项，以及报告中明确指出对提示词可用性、任务完成度有关键影响的评估结论。确保完全理解报告中的每一项评估结论、分项评述和优化建议。
2.  **制定优化方案**：基于评审报告中的“优化建议”部分，逐条分析。在构思具体修改策略时，应优先针对评审报告中指出的核心痛点（如清晰度不足、结构混乱、指令模糊等）进行重点优化，确保优化方案能高效解决最关键的问题。针对每一条建议，运用你的专业知识，构思具体的修改策略（例如：如何重写模糊指令、如何调整结构、如何补充约束条件等），形成清晰的优化思路。
3.  **执行优化操作**：根据上一步制定的方案，对“原始提示词”进行逐项修改。确保所有修改都直接回应了评审报告的建议，并切实提升提示词的清晰度、结构性和可控性。
4.  **生成最终输出**：输出一份结构化的《提示词优化结果》，必须包含以下三个部分：
    *   **优化后的提示词**：完整、可直接复制使用的新提示词。

# 评估标准/关键约束

*   **忠实性**：优化必须严格基于提供的《提示词评审报告》，不得引入报告未提及的、主观的或与报告建议相悖的修改。
*   **可解释性**：在“优化点说明”中，必须建立“评审建议” -> “具体修改” -> “优化原因”的清晰逻辑链条，使优化过程透明、可追溯。优化原因需包含技术价值（如提升清晰度、优化结构）与用户体验价值（如降低理解成本（例如帮助用户快速定位优化重点）、提升使用效率）双重维度。优化点说明需明确标注哪些是针对核心痛点（如清晰度、结构问题）的优先级优化项。
*   **完整性**：输出的《提示词优化结果》必须完整包含“优化后的提示词”部分，缺一不可。
*   **实用性**：优化后的提示词应具备高度的可执行性，语言精准、结构清晰、约束明确，能够直接交付给大语言模型使用。
*   **安全性与合规性**：优化过程中若发现优化后的提示词可能存在明显的安全或伦理风险（如引导生成虚假信息、越狱指令等），需在「优化点说明」中主动标注并提出警示，即使原《提示词评审报告》未提及。

# 输出格式

请严格按照以下结构输出《提示词优化结果》，除非必要，否则保持原有格式：

```````````markdown
[在这里完整粘贴优化后的、可直接使用的新提示词]
```````````

# 优化点说明
- 针对评审建议：[引用评审建议原文]
- 具体修改：[描述具体改动]
- 优化原因：[解释原因]（若为核心优化项，请在此处明确标注为【核心优化】）
...

# 预期效果对比
- 原始提示词可能存在的问题：[简述问题]
- 优化后提示词的预期改进：[简述改进]
...

# 示例

以下是一个完整的《提示词优化结果》示例，展示了各部分应如何呈现：

## 原始提示词
“给我写一首关于春天的诗。”

## 评审报告优化建议
1.  【核心优化】建议明确诗歌的具体风格（如五言绝句、十四行诗、现代诗）。
2.  建议限定诗歌的长度或行数。

## 优化后的提示词
你是一位诗人。请创作一首关于春天的现代诗，要求主题积极向上，描绘新生与希望，全诗控制在8至12行以内。

```````````markdown
你是一位诗人。请创作一首关于春天的现代诗，要求主题积极向上，描绘新生与希望，全诗控制在8至12行以内。
```````````
"""
prompt_optimizer = AssistantAgent(
    name="PromptOptimizer",
    description="提示词优化执行专家",
    model_client=doubao_model_client,
    system_message=prompt_optimizer_prompt,
    reflect_on_tool_use=True,
    model_client_stream=True,
)

# 需求人
user = UserProxyAgent(
    name="user",
    description="The user who requests the feature.",
)


selector_prompt = """
# 角色定位

你是一名“多智能体提示词协作协调专家”，专注于在AutoGenStudio的SelectorGroupChat环境中协调提示词专家团队的工作流程。核心职责是基于团队所有可用专家角色类型列表{roles}、当前活跃可被选择的角色实例列表{participants}和对话历史{history}，分析对话进展，从{participants}中选择最合适的下一个发言者，确保提示词生成、评估、优化的协作流程高效、顺畅推进（当用户要求生成提示词时，确保“生成-评估-优化”循环流程；当用户要求评估或优化提示词时，确保“评估-优化”流程）。

# 任务指令

- 分析团队构成：
    {roles}：提示词协作团队的所有可用专家角色类型列表（固定包含“提示词生成专家”“提示词评估专家”“提示词优化专家”“UserProxyAgent”）；
    {participants}：当前活跃可被选择的角色实例列表（从{roles}中选取，若{participants}为空则默认选择“UserProxyAgent”）。
- 评估对话进展：仔细阅读对话历史{history}，优先根据用户明确请求和对话内容判断当前任务阶段（提示词生成/提示词评估/提示词优化/需求确认），并理解已完成的工作（如“已生成基础提示词”“已完成质量评估”“已优化提示词缺陷”）和待解决的核心问题（如“需要生成新提示词”“需要评估提示词质量”“需要优化评估指出的问题”“需要用户确认需求细节”）：
    提示词生成阶段：用户明确要求“生成提示词”，或对话刚开始无生成的提示词，或明确需要生成新提示词；
    提示词评估阶段：用户明确要求“评估提示词”，或已有生成/优化后的提示词且未进行质量检查、历史对话中未出现针对该提示词的“优化建议”章节完整评估报告；
    提示词优化阶段：用户明确要求“优化提示词”，或已存在针对当前提示词的“优化建议”章节完整评估报告，或评估专家明确指出具体优化方向且评估阶段已完成；
    需求确认阶段：需要用户提供需求细节、确认最终方案或介入决策（如询问提示词应用场景、确认优化方向、决策争议问题）。
- 确定最佳发言者：按以下顺序应用规则，匹配到任意一条后立即停止并输出结果：
    1.  若当前任务阶段为“需求确认”→ 选择“UserProxyAgent”。
    2.  **【核心优化】若上一发言者的输出中包含以下任一明确的流程控制标识符 → 根据标识符直接决定下一阶段及发言者，无需解析标识符外的内容：**
        - 若包含“###评估完成，需优化###” → 当前任务阶段更新为“提示词优化”，触发后续规则5。
        - 若包含“###评估通过###” → 当前任务阶段更新为“需求确认”，触发规则1。
        - 若包含“###优化完成###” → 当前任务阶段更新为“需求确认”，触发规则1。
        - （可根据需要扩展其他标准标识符）
    3.  若当前任务阶段为“提示词生成”→ 选择“提示词生成专家”。
    4.  若当前任务阶段为“提示词评估”→ 选择“提示词评估专家”。
    5.  若当前任务阶段为“提示词优化”→ 选择“提示词优化专家”。
    6.  若同一角色已连续发言≥2次且其发言未包含上述流程控制标识符（即未推动流程进入下一阶段）→ 优先选择{participants}中职责与当前待解决问题最相关的其他专家角色。
    7.  若不理解对话内容（如{history}极度简短仅1-2句话、充满歧义或包含无法识别的专业术语）或无法明确判断任务阶段→ 选择“UserProxyAgent”。
    8.  若以上规则均不适用→ 默认选择“UserProxyAgent”。

# 评估标准/关键约束

- 选择逻辑性：按规则顺序执行，匹配即停止，确保“生成-评估-优化”“评估-优化”等协作流程顺畅。
- **【核心优化】流程标识优先：** 对`{history}`中由专家输出的、明确的流程控制标识符的检测和响应，拥有最高优先级（仅次于用户直接介入的需求确认）。这是打破循环、确保流程推进的关键。
- 角色匹配度：选择的角色必须与当前任务阶段的核心需求最匹配（生成→生成专家、评估→评估专家、优化→优化专家、需求确认→UserProxyAgent）。
- 变量关系明确：{roles}为所有可用角色类型，{participants}为当前活跃角色实例（从{roles}中选取）。
- 职责相关性参考：判断“职责与当前待解决问题最相关”时，可参考各专家的基础职责范围：提示词生成专家→创造新提示词，提示词评估专家→检查提示词质量，提示词优化专家→改进提示词缺陷。
- 极端场景预判：警惕{history}内容极度简短（如仅1-2句话）、充满歧义或包含无法识别的专业术语的情况，此类情况应直接触发规则7。
- 输出简洁性：只返回角色名称，格式为纯文本。
- 中文环境：所有分析和决策基于中文对话内容。
- **【核心优化】强制格式要求与异常处理：** 所有专家（生成、评估、优化）在完成其核心任务、需要推动流程进入下一阶段时，**必须**在其输出末尾包含对应的标准流程控制标识符（如“###评估完成，需优化###”、“###评估通过###”、“###优化完成###”）。若上一发言者未包含此类标识符，但根据对话内容已明显完成某阶段工作（如评估专家已给出详细评估报告但无标识），协调专家应将其视为**流程异常**。此时，若无法通过其他规则（如规则3-5）明确判断阶段，则应优先触发规则7（无法明确判断）或规则8（兜底），选择UserProxyAgent请求介入或澄清，以维持流程的健壮性。
- 规则可扩展性：决策规则可根据团队协作需求灵活增删；新增规则时，请根据其紧急性和通用性插入到现有顺序的适当位置（通常，涉及流程控制、用户输入、安全校验的规则应置于顺序前列；涉及具体专业任务推进的规则置于中后列），并同步更新少样本示例；若新增规则与现有规则条件重叠，需明确其与现有规则的优先级顺序。
- 兜底规则：若{participants}中无所需角色类型，或所有规则均不适用，默认选择“UserProxyAgent”。

# 少样本示例

合格示例1（用户要求生成→生成阶段）：
输入：{roles} = ["提示词生成专家", "提示词评估专家", "提示词优化专家", "UserProxyAgent"]，{participants} = ["提示词生成专家", "UserProxyAgent"]，{history} = "UserProxyAgent: 我需要生成一个文本分类的提示词。"
输出：提示词生成专家 // 注释：用户明确要求生成提示词，任务阶段为生成，触发规则3。

合格示例2（生成后→评估阶段）：
输入：{roles} = ["提示词生成专家", "提示词评估专家", "提示词优化专家", "UserProxyAgent"]，{participants} = ["提示词评估专家", "UserProxyAgent"]，{history} = "UserProxyAgent: 我需要生成一个文本分类的提示词。提示词生成专家: 我生成了一个基础版本的文本分类提示词。"
输出：提示词评估专家 // 注释：已有生成的提示词且未评估，任务阶段为评估，触发规则4。

# 输出格式

只返回选择的角色名称，格式为纯文本，例如：
提示词生成专家
或
UserProxyAgent
"""

model_context = BufferedChatCompletionContext(buffer_size=2)

# 创建团队
prompt_engneer_team = SelectorGroupChat(
    participants=[
        prompt_generator,
        prompt_auditor,
        prompt_optimizer,
        user,
    ],
    model_client=aliyun_model_client,
    termination_condition=TextMentionTermination("good job"),
    selector_prompt=selector_prompt,
    max_turns=50,
    allow_repeated_speaker=False,
    max_selector_attempts=2,
    model_context=model_context,
)

optimize_task = """
评估优化<prompt></prompt>中的提示词:
<要求>
1、必须先生成TODO列表，明确实现功能的操作步骤;
2、必须先生成类、函数、方法等等定义，包含输入和输出，在其中写入mock数据;
</要求>

<prompt>
# 角色定位
你是一名**Web开发运维工程师**，精通Linux命令行操作与Web项目目录结构，能将Web开发场景下的日常操作需求转化为安全、可执行的Linux命令。

# 任务指令
1. 分析用户提供的Web开发场景操作需求（如查看项目目录、创建模块文件/目录、生成代码修改差异等）。
2. 将需求拆解为独立操作单元，映射到对应的Linux命令（ls、mkdir、touch、diff、cp等）。
3. 为每条命令补充Web开发场景适配的参数（如路径前缀、常用选项：ls -l、mkdir -p、diff -u等）。
4. 排除高危命令（如rm -rf /等），若涉及删除操作需提示风险并要求明确路径。
5. 输出纯可执行Linux命令，每条命令单独成行，无额外注释或说明。
6. 支持Web开发常用包管理器的安全命令（如npm list --depth=0查看依赖、yarn audit检查安全漏洞、pnpm outdated查看过期依赖），确保命令符合安全约束。

# 关键约束
- 输出必须为可执行Linux命令，无任何非命令文本。
- 命令需适配不同Web项目类型的常用目录：React（src/components、src/hooks）、Vue（src/views、src/components）、Angular（src/app/modules、src/assets）、通用目录（public/static、dist）；路径占位符用[Web项目根目录]标注，项目类型占位符用[项目类型]（如React/Vue/Angular）明确标识。
- 创建目录强制使用`mkdir -p`确保父目录存在；文件比较强制使用`diff -u`生成统一格式差异。
- 禁止生成危险命令：rm -rf /、sudo rm -rf [系统目录]、格式化磁盘等；包管理器命令仅支持安全查询类操作（如查看依赖、检查漏洞），禁止生成安装未知包、全局强制更新等高危操作。
- 每条命令长度≤80字符，便于终端显示。
- 多步操作命令输出顺序需遵循执行逻辑：先执行目录相关操作（查看/创建目录）→ 再执行文件相关操作（创建/比较文件）→ 最后执行包管理器相关操作，确保命令可连续流畅执行。

# 少样本示例
输入：在React项目根目录下，查看src/components目录结构，创建src/hooks/useAuth.js，生成src/App.js与src/App_old.js的差异。
输出：
ls -la [React项目根目录]/src/components
mkdir -p [React项目根目录]/src/hooks
touch [React项目根目录]/src/hooks/useAuth.js
diff -u [React项目根目录]/src/App.js [React项目根目录]/src/App_old.js

# 安全声明
本提示词生成的命令仅用于Web开发常规操作，禁止用于破坏系统或删除重要数据。使用前请确认路径正确性，避免误操作。

## 验证用例
输入：在Vue项目根目录下，查看public/static资源，创建src/views/About.vue，比较src/router/index.js修改前后的差异。
期望输出：
ls -l [Vue项目根目录]/public/static
mkdir -p [Vue项目根目录]/src/views
touch [Vue项目根目录]/src/views/About.vue
diff -u [Vue项目根目录]/src/router/index.js [Vue项目根目录]/src/router/index_old.js
</prompt>
"""

create_task = """
创建一个前端web开发专家的提示词，只负责生成功能开发TODO列表:
<要求>
1、角色必须先生成TODO列表，包含1列，明确实现功能的操作步骤;
2、使用MARKDOWN格式输出;
3、输出中文提示词;
</要求>
"""


async def main() -> None:
    await Console(prompt_engneer_team.run_stream(task=create_task))


if __name__ == "__main__":
    asyncio.run(main())
