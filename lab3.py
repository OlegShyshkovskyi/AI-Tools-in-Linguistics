from openai import OpenAI
import os

client = OpenAI(
    api_key="AIzaSyBSCqrLDflFpxHPTTVCby-hOK22JsaJDFg", 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
messages = [
    {"role": "system", "content": "Ти — корисний, розумний та ввічливий штучний інтелект. Відповідай українською мовою."}
]

print("Чат-бот запущено! (Напишіть 'вихід' для завершення)")
while True:
    user_input = input("\nВи: ")
    
    if user_input.lower() in ['вихід', 'exit', 'quit']:
        print("Чат-бот: До побачення!")
        break
        
    messages.append({"role": "user", "content": user_input})
    
    try:
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=messages,
            temperature=0.7
        )
        
        bot_reply = response.choices[0].message.content
        print(f"Чат-бот: {bot_reply}")
        
        messages.append({"role": "assistant", "content": bot_reply})
        
    except Exception as e:
        print(f"\n[Помилка]: Не вдалося отримати відповідь. Деталі: {e}")