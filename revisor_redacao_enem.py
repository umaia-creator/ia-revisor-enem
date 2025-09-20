import streamlit as st
import language_tool_python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import MarianMTModel, MarianTokenizer
# ... outras importações ...

# =====================
# Configuração
# =====================
st.set_page_config(page_title="Ajuste Fino", page_icon="📝", layout="centered")

# --- Adicione esta linha para o logo ---
st.image("logo_ajuste_fino.png", width=200) # Ajuste a largura conforme necessário
# ---------------------------------------

st.title("✍️ Ajuste Fino") # Este já é o título que você colocou

# ... o restante do seu código ...

# =====================
# Configuração
# =====================
st.set_page_config(page_title="Ajuste Fino", page_icon="📝", layout="centered")
st.title("✍️ Ajuste Fino")

# LanguageTool (correção PT-BR)
tool = language_tool_python.LanguageTool("pt-BR")

# ======= Carregar modelo LLaMA (ou outro compatível) =======
# Exemplo: "meta-llama/Llama-2-7b-hf" -> precisa de GPU forte
# Para CPU fraco, use "EleutherAI/gpt-neo-125M"
LLAMA_MODEL = "EleutherAI/gpt-neo-125M"

@st.cache_resource
def carregar_llama(model_name=LLAMA_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # device=-1 = CPU
    return gen

with st.spinner("Carregando modelo LLaMA (pode demorar na primeira vez)..."):
    llama_pipeline = carregar_llama()

# ======= Tradução (MarianMT) =======
TRANSLATION_MODELS = {
    "en": "Helsinki-NLP/opus-mt-pt-en",
    "es": "Helsinki-NLP/opus-mt-pt-es",
    "fr": "Helsinki-NLP/opus-mt-pt-fr",
    "de": "Helsinki-NLP/opus-mt-pt-de",
    "it": "Helsinki-NLP/opus-mt-pt-it",
}

@st.cache_resource
def carregar_mariana(model_name):
    tok = MarianTokenizer.from_pretrained(model_name)
    mod = MarianMTModel.from_pretrained(model_name)
    return tok, mod

def traduzir_mariana(texto, idioma_destino="en"):
    if idioma_destino not in TRANSLATION_MODELS:
        return "Idioma não suportado."
    model_name = TRANSLATION_MODELS[idioma_destino]
    tok, mod = carregar_mariana(model_name)
    batch = tok.prepare_seq2seq_batch([texto], return_tensors="pt")
    translated = mod.generate(**{k: v for k, v in batch.items()})
    tgt = [tok.decode(t, skip_special_tokens=True) for t in translated]
    return tgt[0]

# ======= Funções =======
def corrigir_com_languagetool(texto):
    erros = tool.check(texto)
    texto_corrigido = language_tool_python.utils.correct(texto, erros)
    return texto_corrigido, erros

def revisar_com_llama(texto):
    prompt = (
        "Você é um corretor de redações estilo ENEM. "
        "Leia o texto abaixo e retorne:\n"
        "1) Versão revisada com melhor clareza, coesão e gramática.\n"
        "2) Sugira 5 conectivos adequados.\n"
        "3) Faça 2-3 comentários curtos sobre a estrutura.\n\n"
        f"Texto:\n{texto}\n\nResposta:"
    )
    saida = llama_pipeline(
        prompt,
        max_new_tokens=256,   # controla o tamanho da resposta
        do_sample=True,
        top_p=0.9,
        temperature=0.7,      # opcional, dá mais variação
        num_return_sequences=1
    )
    return saida[0]["generated_text"]


# =====================
# Interface Streamlit
# =====================
texto_usuario = st.text_area("Cole sua redação:", height=300)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("✅ Corrigir (LanguageTool)"):
        if texto_usuario.strip():
            with st.spinner("Corrigindo texto..."):
                texto_corrigido, erros = corrigir_com_languagetool(texto_usuario)
            st.subheader("Texto Corrigido")
            st.write(texto_corrigido)
            if erros:
                st.subheader("Erros encontrados")
                for e in erros[:20]:
                    st.write(f"- {e.ruleId}: {e.message}")

with col2:
    if st.button("🔎 Revisar (LLaMA)"):
        if texto_usuario.strip():
            with st.spinner("Gerando revisão com LLaMA..."):
                revisao = revisar_com_llama(texto_usuario)
            st.subheader("Revisão (LLaMA)")
            st.write(revisao)

with col3:
    idioma = st.selectbox("Traduzir para:", ["en", "es", "fr", "de", "it"])
    if st.button("🌍 Traduzir"):
        if texto_usuario.strip():
            with st.spinner("Traduzindo..."):
                traducao = traduzir_mariana(texto_usuario, idioma)
            st.subheader(f"Tradução ({idioma})")
            st.write(traducao)

st.markdown("---")
st.subheader("🔗 Conectivos úteis (extra)")
st.write("- Causa: porque, devido a, em razão de")
st.write("- Consequência: portanto, assim, por conseguinte")
st.write("- Contraste: contudo, entretanto, por outro lado")
st.write("- Exemplificação: por exemplo, como ilustrado por")
st.write("- Conclusão: em suma, finalmente, para concluir")


