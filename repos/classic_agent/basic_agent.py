from multiprocessing import Value
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

class HelloAgentsLLM:
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        
        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 1) -> str:
        print(f"calling {self.model} ...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            collected_content: List[str] = []

            for chunk in response:
                # 有些 chunk 可能没有 choices，或者 choices 为空，先判断一下
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                # 有些 chunk 只更新 role / finish_reason，没有 content
                if delta is None or delta.content is None:
                    continue

                content = delta.content
                print(content, end="", flush=True)
                collected_content.append(content)

            print()
            return "".join(collected_content)
        except Exception as e:
            # 这里最好返回字符串，否则上面的类型标注会“说一套做一套”
            print("Error in think():", repr(e))
            return f"Error: {e}"

if __name__ == "__main__":
    try:
        llmClient = HelloAgentsLLM()

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        responseText = llmClient.think(exampleMessages)
        if responseText:
            print(responseText)
    except ValueError as e:
        print(e)