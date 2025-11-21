from cmd import PROMPT
from email import message
from typing import List, Dict, Any, Optional
from basic_agent import HelloAgentsLLM
class Memory:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        record = {"type": record_type, "content": content}
        self.records.append(record)
        print (f"memory updated, add {record_type} record")

    def get_trajectory(self) -> str:
        trajectory_parts = []
        for record in self.records:
            if record["type"] == 'execution':
                trajectory_parts.append(f"---last attempt code ---\n{record['content']}")
            elif record["type"] == 'reflection':
                trajectory_parts/append(f"---critic feedback---\n{record["content"]}")
            
        return "\n\n".join(trajectory_parts)

    def get_last_execution(self) -> Optional[str]:
        for record in reversed(self.records):
            if record["type"] == 'execution':
                return record['content']
        return None

INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释。
"""

REFLECT_PROMPT_TEMPLATE = """
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下Python代码，并专注于找出其在<strong>算法效率</strong>上的主要瓶颈。

# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

请分析该代码的时间复杂度，并思考是否存在一种<strong>算法上更优</strong>的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。
如果代码在算法层面已经达到最优，才能回答“无需改进”。

请直接输出你的反馈，不要包含任何额外的解释。
"""

REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务:
{task}

# 你上一轮尝试的代码:
{last_code_attempt}
评审员的反馈：
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""

class ReflectionAgent:
    def __init__(self, llm_client, max_iteration=3):
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iteration = max_iteration

    def run(self, task: str):
        print ("---start---")

        print ("---initial try---")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        messages = [{"role": "user", "content": initial_prompt}]
        initial_code = self.llm_client.think(messages=messages)
        self.memory.add_record("execution", initial_code)

        for i in range(self.max_iteration):
            print (f"{i+1}/{self.max_iteration} iterations")

            print ("reflecting")
            last_code = self.memory.get_last_execution()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(task=task, code=last_code)
            messages = [{"role": "user", "content": reflect_prompt}]
            feedback = self.llm_client.think(messages=messages)
            self.memory.add_record("reflection", feedback)

            if "无需改进" in feedback:
                print ("no additional improvement needed")
                break
            
            print ("optimizing")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task = task,
                last_code_attempt = last_code,
                feedback = feedback
            )
            messages = [{"role": "user", "content": refine_prompt}]
            refined_code = self.llm_client.think(messages=messages)
            self.memory.add_record("execution", refined_code)
        final_code = self.memory.get_last_execution()
        print (f"task done, final code {final_code}")

if __name__ == "__main__":
    llm_client = HelloAgentsLLM()

    agent = ReflectionAgent(llm_client, max_iteration=3)

    task = "编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。"
    agent.run(task)