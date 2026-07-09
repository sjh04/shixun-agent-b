"""构建 B5 检索评测语料（RQ1）。

生成 data/memory_eval/corpus/ 下的记忆库（global/ + conversations/ + memory_index.json）
和 data/memory_eval/corpus_queries.json 标注查询集。

语料设计（对应 proposal 6.4 节）：
- 24 条多主题记忆（6 全局 + 18 对话），importance / 时间戳各不相同；
- 2 条长文档（mem_conv_c05 / mem_conv_c11），关键事实埋在中段，用于验证 chunk 召回；
- 1 组三因子对照（mem_conv_c03 新结论 vs mem_conv_c04 过时结论），
  旧文档故意堆更多查询词，纯相关度会排错，需要时近性 × 重要性纠正；
- 若干含英文错误码的记忆（keyword 检索占优）与纯中文改述查询（向量/HyDE 占优）。

用法（在 code/ 目录下）：
    python build_b5_eval_corpus.py
"""
from __future__ import annotations

from pathlib import Path

from common.io_utils import write_json, write_text


CORPUS_ROOT = Path(__file__).resolve().parents[1] / "data" / "memory_eval" / "corpus"
QUERIES_PATH = Path(__file__).resolve().parents[1] / "data" / "memory_eval" / "corpus_queries.json"


def _pad(paragraphs: list[str]) -> str:
    return "\n\n".join(paragraphs)


LONG_DOC_C05 = _pad(
    [
        "这次对话完整记录了 B1 到 B5 五个模块第一次联调的全过程，问题比预想的多，前后花了一个下午加一个晚上。"
        "最开始的现象是 B1 编排循环在第二轮工具调用后直接退出，终端没有任何报错信息，final_answer 也是空的，"
        "trace.json 里的 status 停在 running，看起来像是流程走到一半被静默吞掉了。为了定位问题，我们先把"
        "五个模块的日志级别全部调到最详细，重新跑了一遍，逐条对比每一轮的输入输出。",
        "第一步排查消息编排。把 trace.json 里的每轮 AIMessage 和 ToolMessage 打印出来对比，发现第二轮的"
        "工具参数 JSON 里混入了模型输出的前导说明文字，导致 B3 工具层解析失败，而这个异常被上层的兜底逻辑"
        "吞掉了，没有回传给 B1。给 B3 的参数解析加了 strip 处理和花括号定位之后，这个问题彻底消失。"
        "顺带给异常路径补了结构化错误返回，以后再出解析问题至少能在 trace 里看到 errors 记录，"
        "不会再出现无声失败。",
        "第二步是 B4 的输出格式。prompt_json 模式要求模型只输出一个 JSON 对象，但 4B 模型偶尔会在 JSON"
        "前面加一句中文解释，比如'好的，我将调用工具'。调整了 system prompt，明确要求禁止输出任何 JSON"
        "以外的字符，并在解析侧做了宽松提取。问题明显缓解但没有根除，大约二十次里还会出现一次，"
        "后续正式演示前需要多跑几次确认稳定性，或者考虑在解析失败时自动重试一轮。",
        "第三步排查记忆注入。B5 返回的 selected_memory.json 里 content 字段是完整的，检索本身没有问题，"
        "但 B1 注入 messages 时把 <memory> 标签放在了 system prompt 之前，导致模型在长上下文下偶尔"
        "完全忽略记忆内容，回答里引用不到历史结论。把注入位置调整到 system prompt 之后、用户输入之前，"
        "再测记忆引用就稳定了。这个顺序问题很隐蔽，值得写进接口文档。",
        "中间还遇到一个环境问题：服务器上另一个同学在跑训练任务，两张卡的显存各被占了一大半，"
        "我们的模型加载直接失败，报的还是一个容易误导人的设备映射错误而不是显存不足。"
        "等到晚上卡空出来才继续联调。这提醒我们演示前一定要先看 nvidia-smi 确认显存余量，"
        "共享服务器上这一步不能省。",
        "结论（重要）：正式跑 run_full_demo.py 全系统演示之前必须做三件事——第一，确认 outputs/full_demo"
        "目录存在且当前用户可写，否则最后的落盘环节会失败，前面几分钟全部白跑；第二，命令行参数"
        " --llm_mode 必须显式设为 prompt_json，不传的话脚本会静默回退到 mock 模式，演示出来的答案是"
        "预置的假数据，评委一眼就能看穿；第三，先单独跑一次 b4_local_agent_llm.py 冒烟测试，确认显卡"
        "可用、模型能正常加载之后，再开始完整的全系统演示。这三条按顺序执行，缺一不可。",
        "之后又验证了保存路径。run_full_demo 跑完后 memory/conversations/conv_001.md 会被覆盖更新，"
        "memory_index.json 同步刷新对应条目的 updated_at 和摘要，这是预期行为。但重复演示前要想清楚"
        "是否接受覆盖：如果上一轮的对话记忆还要用来展示跨对话召回，就先把 conversation_id 换掉，"
        "避免把演示素材冲掉。",
        "关于耗时的实测数据：完整一轮联调（含模型加载）大约六分钟，其中模型加载约九十秒，两轮工具调用"
        "各三十秒左右，B5 记忆检索在缓存命中时不到一秒、冷启动重算向量时十几秒，剩下的时间花在落盘和"
        "报告生成。正式演示时建议提前把模型加载好放在后台，把观众等待时间压到两分钟以内。",
        "还有一个小插曲：中途 outputs 目录被之前的失败运行留下的半截文件占着，新一轮运行时 JSON 合并"
        "报了奇怪的键冲突。清空输出目录重跑后正常。结论是每次正式演示前先清一遍输出目录，"
        "或者给每次运行加时间戳子目录。",
        "最后把所有修复点整理进了各模块的 README 小节，B3 的解析补丁和 B1 的注入位置调整都已提交仓库。"
        "下次联调的重点是验证记忆跨对话复用的效果：第一轮对话沉淀的结论，第二轮换个问法还能不能被"
        "检索出来并影响回答。",
    ]
)

LONG_DOC_C11 = _pad(
    [
        "这轮对话专门调 Qwen3.5-4B 的生成参数，目标是让 B4 的输出既稳定又可复现，把全部尝试过程记录下来"
        "供后面的同学参考。测试方法是固定五个典型任务输入（两个工具调用、两个纯问答、一个长文总结），"
        "每改一个参数就把五个输入各跑三遍，统计 JSON 格式错误率、答案长度和端到端延迟。",
        "先试温度。temperature 从默认的 0.7 降到 0.3 再降到 0，工具调用输出的 JSON 格式错误率从大约"
        "五分之一降到二十分之一再降到几乎为零。原因很直观：采样温度越高，模型越容易在结构化输出里"
        "夹带自由发挥的文字。结论是决策类调用一律用贪心解码，不给采样留口子。"
        "创作类任务如果以后有需求再单独开温度，和决策链路隔离。",
        "然后是 max_new_tokens。一开始给了 2048，观察输出发现模型经常把工具返回的结果原样复述一遍，"
        "再加一段车轱辘话总结，白白浪费生成配额还拖慢响应。截到 1024 之后，五个测试输入都没有出现"
        "答案被截断的情况，平均延迟降了接近一半。再往下压到 512 时长文总结任务开始截尾，"
        "所以 1024 是当前任务集下的合理值。",
        "中间试过 top_p 和 top_k 的各种组合，后来意识到在 do_sample 关闭的情况下这些参数根本不生效，"
        "写在配置里只会误导后来的人以为它们在起作用，全部从 model.yaml 里清掉了。"
        "配置文件里只保留真正生效的参数，这是这次调参的一个附带教训。",
        "关键发现（重要）：模型总在正式回答前输出一大段思考过程，各种在 prompt 里写'不要解释、"
        "直接给答案''禁止输出推理过程'都压不住，模型嘴上答应实际照旧。最终确认必须在"
        " tokenizer.apply_chat_template 调用里显式传 enable_thinking=False，才能真正关闭 <think>"
        " 思考链输出；光靠提示词约束是没有用的，这是模板层的开关不是指令层的问题。同时保持"
        " do_sample=False，同一输入的输出才能逐字符一致，后面的消融实验才有可复现的基础。",
        "顺带测了输入长度上限。max_input_tokens 设 4096，超长的记忆上下文会被 B1 侧截断后再进模型，"
        "所以 B5 压缩层的摘要质量直接决定 B4 能看到什么信息，这两层必须配合着调：压缩预算给太小，"
        "模型拿到的记忆残缺不全；给太大又挤占用户输入和工具结果的空间。目前 2000 字符的记忆预算"
        "配 4096 的输入上限是平衡点。",
        "还对比了 bfloat16 和 float16 两种半精度：bfloat16 在这张卡上连续跑完整个测试集没有出现"
        "数值溢出警告，float16 在长文总结任务上偶尔出现，虽然没直接影响输出质量，但保险起见"
        "配置里锁定 bfloat16。显存占用两者几乎一样，没有理由冒险。",
        "另外记录一个容易踩的坑：换了 transformers 版本后 apply_chat_template 的默认行为可能变化，"
        "升级依赖后要重新核对模板渲染出来的完整 prompt 文本，我们在 outputs 里保留了 prompt_text"
        " 落盘就是为了这个。",
        "把以上参数固化进 configs/model.yaml 之后，连续跑了十次同一输入，十次输出完全一致，"
        "生成侧的可复现性问题到此收口。后续如果换模型或者换卡，按同样的测试流程重新过一遍这份"
        "清单即可，预计半小时能跑完。",
    ]
)


MEMORIES: list[dict] = [
    # ---------------- 全局记忆（6 条） ----------------
    {
        "memory_id": "mem_g_arch",
        "memory_type": "global",
        "title": "B1–B5 系统架构分工",
        "summary": "五个模块的职责边界：B1 编排、B2 技能、B3 工具层、B4 决策、B5 记忆。",
        "importance": 9,
        "created_at": "2026-06-10T09:00:00+00:00",
        "body": "B1 负责运行与消息编排，是唯一的入口，维护 messages 与 trace 并按轮次调度其余模块。"
        "B2 提供 Skill 工具函数本体（计算器、文件读取、表格分析、本地搜索、格式转换）。"
        "B3 负责工具说明生成与执行，把 B2 的函数暴露成 schema 并解析模型的调用参数。"
        "B4 封装本地 Qwen3.5-4B 做决策，输入 messages 输出 AIMessage。"
        "B5 是记忆子系统，负责记忆文档的检索注入与保存更新，唯一调用方是 B1。",
    },
    {
        "memory_id": "mem_g_tools",
        "memory_type": "global",
        "title": "工具使用规范",
        "summary": "五个内置工具的输入输出约定与常见误用。",
        "importance": 8,
        "created_at": "2026-06-12T09:00:00+00:00",
        "body": "calculator 接收算式字符串返回数值，内部用高精度计算，不要传自然语言。"
        "file_reader 只能读 data 根目录内的文本文件，路径越界会被拒绝。"
        "table_analyzer 针对 CSV 表格，支持求和、平均值、最大最小值等操作，需要指定列名。"
        "local_file_search 支持通配符模式匹配文件名，可选递归子目录。"
        "format_converter 在 markdown、html、json 等格式间转换文本。所有工具返回统一的"
        "status/errors 结构，调用失败不会中断主流程。",
    },
    {
        "memory_id": "mem_g_env",
        "memory_type": "global",
        "title": "运行环境与依赖版本",
        "summary": "conda 环境 agent 的 Python 与关键依赖版本清单。",
        "importance": 7,
        "created_at": "2026-06-05T09:00:00+00:00",
        "body": "项目统一使用 conda 环境 agent：Python 3.10、torch 2.7.1+cu118、transformers 5.12、"
        "numpy 2.2、PyYAML 6.0。不安装 faiss 和 sentence-transformers，离线环境下模型"
        "local_files_only 加载。新同学入组先 conda activate agent 再跑 requirements.txt 核对版本。",
    },
    {
        "memory_id": "mem_g_schema",
        "memory_type": "global",
        "title": "selected_memory.json 字段约定",
        "summary": "B5 返回给 B1 的记忆注入 JSON 的字段契约。",
        "importance": 7,
        "created_at": "2026-06-15T09:00:00+00:00",
        "body": "B1 的记忆注入只依赖三个基础字段：memory_id、memory_type、content，这三个字段"
        "永远不删不改。检索增强后新增的字段全部是可选的：score 是三因子总分，rank 是最终名次，"
        "factors 拆出相关度、时近性、重要性三个分量，retrieval_mode 记录检索模式，"
        "compressed 与 compression_method 标记压缩情况，flagged 表示被 Poison Gate 标记。"
        "消费方应当忽略不认识的字段，保证向前兼容。",
    },
    {
        "memory_id": "mem_g_prefs",
        "memory_type": "global",
        "title": "用户偏好",
        "summary": "回答语言、产物目录与格式方面的固定偏好。",
        "importance": 6,
        "created_at": "2026-06-01T09:00:00+00:00",
        "body": "所有回答一律使用中文。运行产物统一落盘到 outputs/ 目录下按模块分子目录，"
        "不要写到仓库根目录。金额单位默认人民币。日期用 ISO 格式书写。"
        "长回答优先给结论再给过程。",
    },
    {
        "memory_id": "mem_g_qwen",
        "memory_type": "global",
        "title": "Qwen3.5-4B 加载配置",
        "summary": "本地模型加载的精度、设备映射与安全选项。",
        "importance": 8,
        "created_at": "2026-06-08T09:00:00+00:00",
        "body": "模型权重放在 models/Qwen3.5-4B，加载参数固定为：torch_dtype 用 bfloat16（这张卡上"
        "float16 偶发溢出警告），device_map 设 auto 让权重自动分配到空闲显卡，local_files_only"
        "设 true 保证离线可用，trust_remote_code 设 true。tokenizer 与模型同路径。"
        "加载一次约九十秒，进程内要缓存复用，不要重复加载。",
    },
    # ---------------- 对话记忆（18 条） ----------------
    {
        "memory_id": "mem_conv_c01",
        "memory_type": "conversation",
        "conversation_id": "eval_c01",
        "title": "计算器浮点精度问题",
        "summary": "0.1 加 0.2 不等于 0.3 的原因与 calculator 的处理方式。",
        "importance": 5,
        "created_at": "2026-06-18T10:00:00+00:00",
        "body": "用户问为什么计算器算 0.1 + 0.2 得到 0.30000000000000004。原因是二进制浮点数"
        "无法精确表示十进制小数。calculator 工具内部已改用 Decimal 做十进制运算，"
        "对外返回结果前按需要的小数位四舍五入，所以通过工具计算不会再出现这类尾数。"
        "直接在 Python 里用 float 仍会有该现象，属于语言特性不是 bug。",
    },
    {
        "memory_id": "mem_conv_c02",
        "memory_type": "conversation",
        "conversation_id": "eval_c02",
        "title": "CSV 列平均值统计",
        "summary": "用 table_analyzer 统计 CSV 某列平均值的正确姿势。",
        "importance": 5,
        "created_at": "2026-06-20T10:00:00+00:00",
        "body": "统计 CSV 某一列的平均值要用 table_analyzer 工具：传入表格路径、操作类型 mean、"
        "以及目标列名。列名必须与表头完全一致，含空格时要原样传入。数值列里混有空值时"
        "工具会自动跳过空值再求平均，行为与 Excel 一致。求和用 sum、最大最小用 max/min，"
        "同一次调用只支持一个操作。",
    },
    {
        "memory_id": "mem_conv_c03",
        "memory_type": "conversation",
        "conversation_id": "eval_c03",
        "title": "CUDA OOM 最新解决方案",
        "summary": "显存不足的现行处理办法：bfloat16 + max_memory 上限 + 减小生成长度。",
        "importance": 8,
        "created_at": "2026-06-28T10:00:00+00:00",
        "body": "推理时报 CUDA out of memory 的最新结论：第一，加载用 bfloat16 而不是 float32，"
        "显存直接省一半；第二，给 from_pretrained 传 max_memory 把单卡上限压到 20GiB，"
        "留出生成期激活的余量；第三，把 max_new_tokens 从 2048 降到 1024。"
        "三条一起用之后再没有出现过 OOM。注意：五月份那次结论（用 float32 加小 batch）"
        "已经过时作废，不要再参考。",
    },
    {
        "memory_id": "mem_conv_c04",
        "memory_type": "conversation",
        "conversation_id": "eval_c04",
        "title": "CUDA out of memory 排查记录（旧）",
        "summary": "早期一次显存不足 CUDA out of memory 的排查，结论后来被推翻。",
        "importance": 3,
        "created_at": "2026-05-03T10:00:00+00:00",
        "body": "显卡内存不够、推理报 CUDA out of memory 的排查记录。当时的现象是"
        "加载即崩溃，报错出现在 from_pretrained 阶段。临时结论：把 torch_dtype 改成"
        " float32 并把 batch 压到 1，勉强能跑但极慢；实在不行重启机器碰运气。"
        "（注：这是早期笔记，后来发现 float32 反而更占显存，本结论已被六月末的"
        "新方案取代，仅留档。）",
    },
    {
        "memory_id": "mem_conv_c05",
        "memory_type": "conversation",
        "conversation_id": "eval_c05",
        "title": "全系统第一次联调记录",
        "summary": "B1–B5 第一次联调的排错过程与耗时实测。",
        "importance": 7,
        "created_at": "2026-06-25T10:00:00+00:00",
        "body": LONG_DOC_C05,
    },
    {
        "memory_id": "mem_conv_c06",
        "memory_type": "conversation",
        "conversation_id": "eval_c06",
        "title": "本地文件通配符搜索",
        "summary": "local_file_search 的模式匹配与递归用法。",
        "importance": 4,
        "created_at": "2026-06-14T10:00:00+00:00",
        "body": "要找出目录下所有 markdown 文件，用 local_file_search 传模式 *.md 并打开递归开关，"
        "工具会遍历全部子目录返回相对路径列表。模式匹配只作用于文件名不含路径，"
        "想按路径过滤要在返回结果里自己筛。搜索范围被限制在 data 根目录内，越界返回错误。",
    },
    {
        "memory_id": "mem_conv_c07",
        "memory_type": "conversation",
        "conversation_id": "eval_c07",
        "title": "markdown 转 html 的坑",
        "summary": "format_converter 做 markdown 到 html 转换的注意事项。",
        "importance": 4,
        "created_at": "2026-06-16T10:00:00+00:00",
        "body": "markdown 转 html 用 format_converter，目标格式参数必须显式传 html。"
        "两个坑：一是表格属于扩展语法，简单管道表格能转，嵌套表格会原样输出；"
        "二是行内 HTML 默认会被转义，需要保留标签时加 raw 选项。转出来的 html"
        "是文档片段不含 head，要完整页面得自己包一层。",
    },
    {
        "memory_id": "mem_conv_c08",
        "memory_type": "conversation",
        "conversation_id": "eval_c08",
        "title": "ModuleNotFoundError yaml 解决办法",
        "summary": "No module named yaml 报错的两个原因与修复。",
        "importance": 6,
        "created_at": "2026-06-11T10:00:00+00:00",
        "body": "报 ModuleNotFoundError: No module named 'yaml' 有两个常见原因：一是包名装错，"
        "正确的安装命令是 pip install pyyaml，包名是 pyyaml 而导入名是 yaml，直接"
        " pip install yaml 装到的是另一个废弃包；二是忘了激活 conda 环境，在 base 里"
        "跑了脚本。先 conda activate agent 再确认 pip list 里有 PyYAML。",
    },
    {
        "memory_id": "mem_conv_c09",
        "memory_type": "conversation",
        "conversation_id": "eval_c09",
        "title": "系统提示词模板位置",
        "summary": "system prompt 模板文件路径与替换方式。",
        "importance": 5,
        "created_at": "2026-06-09T10:00:00+00:00",
        "body": "系统提示词模板放在 prompts/ 目录下，具体用哪个由 runtime_input.json 的"
        " system_prompt_path 字段指定，路径相对输入文件所在目录解析。要换提示词"
        "不要直接改默认模板，复制一份改名后把 system_prompt_path 指过去，"
        "保证别人的演示不受影响。模板里的占位符由 B1 在注入时填充。",
    },
    {
        "memory_id": "mem_conv_c10",
        "memory_type": "conversation",
        "conversation_id": "eval_c10",
        "title": "记忆截断问题",
        "summary": "max_memory_chars 预算与超长记忆的压缩策略。",
        "importance": 5,
        "created_at": "2026-06-13T10:00:00+00:00",
        "body": "记忆内容太长被截断的处理：注入预算由 memory.yaml 的 max_memory_chars 控制，"
        "默认 2000 字符，多条记忆按名次先后瓜分预算。单条超预算时不再从中间硬切，"
        "而是调用压缩层生成摘要，保关键信息弃细节；模型不可用时退化为抽取式摘要。"
        "如果发现注入的记忆缺了关键内容，优先调大 max_memory_chars 而不是关压缩。",
    },
    {
        "memory_id": "mem_conv_c11",
        "memory_type": "conversation",
        "conversation_id": "eval_c11",
        "title": "Qwen 生成参数调优记录",
        "summary": "温度、生成长度等参数的完整调优过程与结论。",
        "importance": 7,
        "created_at": "2026-06-26T10:00:00+00:00",
        "body": LONG_DOC_C11,
    },
    {
        "memory_id": "mem_conv_c12",
        "memory_type": "conversation",
        "conversation_id": "eval_c12",
        "title": "BM25 参数讨论",
        "summary": "检索层 BM25 的 k1 与 b 参数取值依据。",
        "importance": 5,
        "created_at": "2026-06-17T10:00:00+00:00",
        "body": "B5 检索层的 BM25 参数定为 k1=1.5、b=0.75，是信息检索文献里的经典默认值。"
        "k1 控制词频饱和速度，中文字符 bigram 的词频分布偏平，1.5 够用；"
        "b 控制文档长度归一化强度，语料里长短文档混杂，0.75 能压住长文档的天然优势。"
        "调参优先级不高，先把 chunk 和融合做好收益更大。",
    },
    {
        "memory_id": "mem_conv_c13",
        "memory_type": "conversation",
        "conversation_id": "eval_c13",
        "title": "poster 排版分工",
        "summary": "结课海报的分工与截止时间。",
        "importance": 2,
        "created_at": "2026-06-30T10:00:00+00:00",
        "body": "海报排版由孙家恒负责整体版式和 B5 板块，其他成员各自供稿自己模块的图和"
        "三句话简介，素材周三前汇总。模板用学院提供的 pptx，图统一导出 300dpi PNG。"
        "打印前留一天余量校对错别字。",
    },
    {
        "memory_id": "mem_conv_c14",
        "memory_type": "conversation",
        "conversation_id": "eval_c14",
        "title": "闲聊记录",
        "summary": "一次与任务无关的闲聊。",
        "importance": 1,
        "created_at": "2026-06-29T10:00:00+00:00",
        "body": "聊了下最近天气太热，机房空调给力所以大家都愿意来实验室。中午食堂二楼"
        "新开的窗口排队太长，建议错峰。没有任何与项目相关的结论。",
    },
    {
        "memory_id": "mem_conv_c15",
        "memory_type": "conversation",
        "conversation_id": "eval_c15",
        "title": "git detached HEAD 恢复",
        "summary": "检出历史提交后回到分支的操作。",
        "importance": 4,
        "created_at": "2026-06-07T10:00:00+00:00",
        "body": "git 提示 detached HEAD 是因为直接 checkout 了某个提交号。想保留当前修改就"
        "用 git switch -c 新分支名 把工作现场变成新分支；不需要保留就 git switch main"
        "直接回主分支。提交在游离状态下做的 commit 不会丢，reflog 里能找回。",
    },
    {
        "memory_id": "mem_conv_c16",
        "memory_type": "conversation",
        "conversation_id": "eval_c16",
        "title": "JSONDecodeError 修复",
        "summary": "Expecting value 报错的根因：输出混入日志前缀。",
        "importance": 6,
        "created_at": "2026-06-19T10:00:00+00:00",
        "body": "json.loads 报 JSONDecodeError: Expecting value: line 1 column 1 的根因是"
        "待解析文本开头混入了模型输出的说明文字或日志前缀，不是 JSON 本身坏了。"
        "修复：解析前先定位第一个左花括号、截取到匹配的右花括号再 loads；"
        "同时在 prompt 里禁止模型输出 JSON 以外的字符。空字符串也会报同样的错，"
        "要先判空给出更明确的错误信息。",
    },
    {
        "memory_id": "mem_conv_c17",
        "memory_type": "conversation",
        "conversation_id": "eval_c17",
        "title": "向量缓存设计",
        "summary": "memory_vector_cache.json 的键设计与失效策略。",
        "importance": 5,
        "created_at": "2026-06-21T10:00:00+00:00",
        "body": "向量缓存文件 memory_vector_cache.json 的作用是避免每次检索都重算 chunk"
        " embedding。缓存键由记忆 id、chunk 序号、向量后端和内容 hash 四部分拼成，"
        "内容一变 hash 就变，旧向量自动失效不会被误用，因此不需要显式清缓存。"
        "qwen 和 hashing 两个后端的向量分开缓存互不覆盖。该文件可随时删除，"
        "代价只是下次检索重算一遍。",
    },
    {
        "memory_id": "mem_conv_c18",
        "memory_type": "conversation",
        "conversation_id": "eval_c18",
        "title": "示例表格字段说明",
        "summary": "data/tables 下演示 CSV 的列含义。",
        "importance": 3,
        "created_at": "2026-06-06T10:00:00+00:00",
        "body": "data/tables 目录下的演示表格各列含义：name 是条目名称，category 是类别标签，"
        "amount 是数值金额（人民币），date 是发生日期 ISO 格式。表头首行固定，"
        "编码统一 UTF-8 无 BOM。做表格分析演示时用这份表，别再造新数据。",
    },
]


# probe 字段标记每条查询的设计意图：
#   keyword_exact      查询含精确错误码/术语，关键词检索应占优
#   paraphrase         查询与目标文档几乎无词面重合，考语义召回（qwen 向量 / HyDE）
#   chunk_buried       答案埋在长文档中段，title/summary 无线索，考 chunk 级召回
#   three_factor       新旧结论冲突，旧文档词面重合更高，考时近性 × 重要性纠偏
#   multi_relevant     多条记忆都相关
QUERIES: list[dict] = [
    {"query": "整个系统里哪个模块负责决定下一步调用什么工具？", "relevant_ids": ["mem_g_arch"], "probe": "paraphrase"},
    {"query": "表格里某一列数字的均值怎么算？", "relevant_ids": ["mem_conv_c02", "mem_g_tools"], "probe": "multi_relevant"},
    {"query": "显卡内存不够、推理时报 CUDA out of memory 该怎么处理？", "relevant_ids": ["mem_conv_c03"], "probe": "three_factor"},
    {"query": "为什么全系统演示跑出来的答案像是预置的假数据？", "relevant_ids": ["mem_conv_c05"], "probe": "chunk_buried"},
    {"query": "报错 ModuleNotFoundError: No module named 'yaml' 怎么解决？", "relevant_ids": ["mem_conv_c08"], "probe": "keyword_exact"},
    {"query": "提示词里写了不要解释，模型还是先输出一大段推理，怎么才能真正关掉？", "relevant_ids": ["mem_conv_c11"], "probe": "chunk_buried"},
    {"query": "模型加载要用什么数值精度，多卡怎么分配权重？", "relevant_ids": ["mem_g_qwen"], "probe": "paraphrase"},
    {"query": "B5 返回给编排模块的结果里，哪些字段是永远不能动的？", "relevant_ids": ["mem_g_schema"], "probe": "paraphrase"},
    {"query": "输出的报告应该保存到哪里、用什么语言写？", "relevant_ids": ["mem_g_prefs"], "probe": "paraphrase"},
    {"query": "想把项目里所有的说明文档一次性找出来怎么做？", "relevant_ids": ["mem_conv_c06"], "probe": "paraphrase"},
    {"query": "转出来的网页里表格显示不对是什么原因？", "relevant_ids": ["mem_conv_c07"], "probe": "paraphrase"},
    {"query": "检索打分函数里的两个超参数是怎么定的？", "relevant_ids": ["mem_conv_c12"], "probe": "paraphrase"},
    {"query": "新同学入组配环境要装哪些版本的依赖？", "relevant_ids": ["mem_g_env"], "probe": "paraphrase"},
    {"query": "为什么 0.1 + 0.2 在程序里不等于 0.3？", "relevant_ids": ["mem_conv_c01"], "probe": "keyword_exact"},
    {"query": "json.loads 报 Expecting value: line 1 column 1 是什么问题？", "relevant_ids": ["mem_conv_c16"], "probe": "keyword_exact"},
    {"query": "注入的历史上下文超出预算会发生什么？", "relevant_ids": ["mem_conv_c10"], "probe": "paraphrase"},
    {"query": "embedding 算出来的结果存在哪里，需要手动清理吗？", "relevant_ids": ["mem_conv_c17"], "probe": "paraphrase"},
    {"query": "git 出现 detached HEAD 状态怎么恢复到分支？", "relevant_ids": ["mem_conv_c15"], "probe": "keyword_exact"},
    {"query": "换一份自定义的 system prompt 应该改哪个文件？", "relevant_ids": ["mem_conv_c09"], "probe": "paraphrase"},
    {"query": "演示数据里的金额是什么货币单位？", "relevant_ids": ["mem_conv_c18", "mem_g_prefs"], "probe": "multi_relevant"},
]


def _memory_markdown(item: dict) -> str:
    heading = "Insight" if item["memory_type"] == "global" else "Final Answer"
    lines = [
        f"# {item['title']}",
        "",
        f"- memory_id: `{item['memory_id']}`",
    ]
    if item.get("conversation_id"):
        lines.append(f"- conversation_id: `{item['conversation_id']}`")
    lines.extend(
        [
            f"- created_or_updated_at: `{item['created_at']}`",
            "",
            f"## {heading}",
            "",
            item["body"],
            "",
        ]
    )
    return "\n".join(lines)


def build() -> None:
    index: dict[str, dict] = {}
    for item in MEMORIES:
        subdir = "global" if item["memory_type"] == "global" else "conversations"
        relative = f"{subdir}/{item['memory_id']}.md"
        write_text(_memory_markdown(item), CORPUS_ROOT / relative)
        index[item["memory_id"]] = {
            "memory_id": item["memory_id"],
            "memory_type": item["memory_type"],
            "title": item["title"],
            "summary": item["summary"],
            "path": relative,
            "conversation_id": item.get("conversation_id"),
            "importance": item["importance"],
            "created_at": item["created_at"],
            "updated_at": item["created_at"],
            "last_accessed_at": item["created_at"],
            "access_count": 0,
        }
    write_json(index, CORPUS_ROOT / "memory_index.json")
    write_json(QUERIES, QUERIES_PATH)
    long_docs = [item["memory_id"] for item in MEMORIES if len(item["body"]) > 1500]
    print(f"corpus: {len(MEMORIES)} memories -> {CORPUS_ROOT}")
    print(f"queries: {len(QUERIES)} -> {QUERIES_PATH}")
    print(f"long docs (chunk probes): {long_docs}")


if __name__ == "__main__":
    build()
