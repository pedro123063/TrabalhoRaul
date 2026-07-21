import time
import re
import ollama
from google import genai

# Mantenha sua chave configurada
API_KEY_GEMINI ="<>" #"CHAVEAQUI"

def send_prompt_to_google(prompt_text: str) -> str:
    # Inicializa o cliente
    client = genai.Client(api_key=API_KEY_GEMINI)

    # 1. Marca o tempo inicial
    start_time = time.time()

    # 2. Faz a chamada para a API
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt_text
    )

    # 3. Marca o tempo final e calcula a diferença
    end_time = time.time()
    elapsed_time = end_time - start_time

    answer_text = response.text

    # 4. Exibe os resultados no console
    print("=" * 60)
    print(f"<pergunta>\n{prompt_text}\n</pergunta>\n")
    print(f"<resposta>\n{answer_text}\n</resposta>\n")
    print(f" Tempo de resposta: {elapsed_time:.2f} segundos")
    print("=" * 60)

    return answer_text

def send_prompt_to_deepseek(prompt_text: str, model_name: str = "deepseek-r1:1.5b") -> str:
    start_time = time.time()
    
    # 1. Faz a chamada local via Ollama
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                'role': 'user',
                'content': prompt_text,
            },
        ]
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    raw_answer = response['message']['content']
    
    # 2. Remove as tags <think>...</think> do DeepSeek-R1 para pegar só o resultado
    answer_text = re.sub(r'<think>.*?</think>', '', raw_answer, flags=re.DOTALL).strip()
    
    print("=" * 60)
    print(f"<pergunta>\n{prompt_text}\n</pergunta>\n")
    print(f"<resposta_deepseek>\n{answer_text}\n</resposta_deepseek>\n")
    print(f" Tempo de resposta (DeepSeek Local): {elapsed_time:.2f} segundos")
    print("=" * 60)

    return answer_text