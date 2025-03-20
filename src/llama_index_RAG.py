from llama_index.llms.ollama import Ollama

llm = Ollama(model="deepseek-r1:7b", request_timeout=320.0)

resp = llm.complete("Who is Paul Graham?")

print(resp)