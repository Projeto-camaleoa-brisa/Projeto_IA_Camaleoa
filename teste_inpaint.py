import requests
import base64

print("🚀 Iniciando teste...")

url = "http://localhost:5001/api/inpaint"

files = {
    "image": open("imagem.png", "rb"),
    "mask": open("mask.png", "rb")
}

data = {
    "prompt": "same shoe, identical shape, same lighting, only change color to red"
}

try:
    response = requests.post(url, files=files, data=data)

    print("📡 Status:", response.status_code)

    result = response.json()

    print("📦 Estrutura:", result.keys())
    print("🔍 DATA COMPLETO:")
    print(result["data"])

    # 🔥 FUNÇÃO QUE PROCURA BASE64 EM QUALQUER LUGAR
    def encontrar_base64(obj):
        if isinstance(obj, str) and obj.startswith("iVBOR"):
            return obj
        elif isinstance(obj, dict):
            for v in obj.values():
                resultado = encontrar_base64(v)
                if resultado:
                    return resultado
        elif isinstance(obj, list):
            for item in obj:
                resultado = encontrar_base64(item)
                if resultado:
                    return resultado
        return None

    base64_str = encontrar_base64(result)

    if not base64_str:
        raise Exception("❌ Não encontrou base64 na resposta")

    print("🧠 Convertendo imagem...")

    img_bytes = base64.b64decode(base64_str)

    with open("resultado.png", "wb") as f:
        f.write(img_bytes)

    print("✅ Imagem salva como resultado.png")

except Exception as e:
    print("❌ ERRO FINAL:", e)