import spacy
import requests
import re
from gensim.models import FastText

# Carregar o modelo de língua do spaCy (garanta que o spaCy esteja instalado)
nlp = spacy.load("en_core_web_sm")

# Baixar um arquivo de texto da internet
url = "https://pt.wikipedia.org/wiki/Immanuel_Kant"  # Exemplo de link para um artigo público (em português)
response = requests.get(url)
text = response.text  # Obter o conteúdo do arquivo como string

# Verificar se o download foi bem-sucedido
if response.status_code == 200:
    print(f"Arquivo baixado com sucesso! Número de caracteres: {len(text)}")
else:
    print("Erro ao baixar o arquivo.")

# Função para limpar o texto removendo tags HTML e símbolos desnecessários
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove caracteres especiais
    text = re.sub(r'\n', ' ', text)  # Remove quebras de linha
    return text

# Função para tokenizar o texto com spaCy
def preprocess_text(text):
    # Limpeza do texto antes da tokenização
    cleaned_text = clean_text(text)
    
    # Usar spaCy para tokenização
    doc = nlp(cleaned_text)
    tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_stop]
    return tokens

# Tokenizar o texto baixado
tokenized_corpus = preprocess_text(text)

# Treinando o modelo FastText com parâmetros ajustados
model = FastText(sentences=[tokenized_corpus], vector_size=300, window=5, min_count=2, workers=4)

# Testando a similaridade entre palavras
similarity = model.wv.similarity('kant', 'philosophy')
print(f"Similaridade entre 'kant' e 'philosophy': {similarity:.4f}")

# Verificando palavras mais semelhantes a 'kant'
similar_words = model.wv.most_similar('kant', topn=5)
print("\nPalavras mais semelhantes a 'kant':")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")
