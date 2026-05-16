

# Motor de Busca Semântica Otimizado para Notícias 

![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-v2.2+-green)
![Status](https://img.shields.io/badge/status-conclu%C3%ADdo-brightgreen)

Este projeto implementa uma pipeline de processamento de linguagem natural (PLN) de alta performance para limpeza textual e busca vetorial semântica em corpo de notícias em português. Ele utiliza representações geométricas em uma hiperesfera para calcular a similaridade de cosseno de forma extremamente veloz e eficiente.

---

## 🧠 Justificativa do Modelo Escolhido

O modelo selecionado para este projeto foi o **`paraphrase-multilingual-MiniLM-L12-v2`** da biblioteca *SentenceTransformers*. A escolha baseou-se nos seguintes pilares:

*   **Suporte Nativo a Múltiplas Línguas (Foco em PT-BR):** Sendo treinado com uma abordagem de destilação de conhecimento multimodal/multilíngue, o modelo preserva a semântica do português brasileiro de forma robusta, entendendo nuances, sinônimos e contextos sem a necessidade de tradução prévia.
*   **Excelente Custo-Benefício Computacional:** Sendo uma versão baseada em MiniLM (com 12 camadas), o modelo entrega embeddings de altíssima qualidade com um consumo de memória e tempo de inferência drasticamente menor do que modelos massivos como BERT-large ou mMARCO. Isso viabiliza a execução mesmo em hardwares limitados (CPU) ou instâncias leves de nuvem.
*   **Foco em Paráfrases e Semântica de Busca:** Ao contrário de modelos puramente auto-regressivos ou de classificação, a série *paraphrase* é otimizada especificamente para mapear textos com significados semelhantes para regiões próximas no espaço vetorial, ideal para responder a consultas de usuários (*queries*) que não usam as palavras exatas do documento original.

---

## 🛠️ Decisões de Arquitetura e Otimizações

*   **Preservação de Capitalização (Case Sensitivity):** Uma alteração crucial documentada no código foi a remoção do `.lower()`. Modelos baseados em Transformers modernos (especialmente os multilíngues) utilizam tokenizadores inteligentes (como WordPiece/SentencePiece) que entendem que siglas (como BCB, IBGE, MDIC) e nomes próprios capitalizados carregam forte teor semântico. Forçar o texto para minúsculo degradaria a precisão do modelo.
*   **Indexação Geométrica na Hiperesfera ($L_2$ Normalization):** Em vez de calcular a similaridade de cosseno tradicional a cada busca (o que exige recalcular normas em tempo de execução), o motor realiza a normalização $L_2$ prévia de todos os embeddings no momento da indexação. Dessa forma, a similaridade de cosseno matemática reduz-se a um simples produto escalar (*dot product*) vetorial:

$$\text{Similaridade} = \mathbf{u} \cdot \mathbf{v}$$

> 💡 *Isto é processado em nível de hardware via `numpy.dot` de forma instantânea.*

*   **Uso de `np.argpartition` em vez de ordenação total:** Em corpora de grande escala, ordenar o array inteiro de similaridades com `argsort` (complexidade $\mathcal{O}(N \log N)$) é um desperdício. O uso do `argpartition` isola os $K$ maiores elementos com complexidade linear $\mathcal{O}(N)$, ordenando estritamente os resultados finais retornados (Top-K).
*   **Sanitização Textual Avançada e Defensiva:** A classe `TextCleaner` foi projetada para lidar com os problemas clássicos de *scrapers* de notícias brasileiros (tags HTML residuais, metadados de data/hora colados ao texto, links quebrados por espaços e barras invertidas de escapes JSON errados).

---

## 🚀 Como Rodar o Projeto

### Pró-requisitos
Certifique-se de ter o Python instalado (versão 3.9 a 3.11 recomendada).

### 1. Clonar ou criar a estrutura de diretórios
O script espera uma pasta chamada `dados/` no mesmo nível do arquivo de execução.

```text
meu-projeto/
├── dados/
│   └── noticias_brutas.json
└── main.py

2. Instalar as Dependências
Instale as bibliotecas necessárias utilizando o pip:

<img width="1057" height="149" alt="image" src="https://github.com/user-attachments/assets/24c1f220-bede-462b-bc08-5131752416c4" />
⚠️ Nota: Se você possuir uma GPU NVIDIA, certifique-se de instalar a versão do PyTorch com suporte ao CUDA para acelerar a vetorização.

3. Formato do Arquivo de Entrada (dados/noticias_brutas.json)
O arquivo JSON de entrada deve conter uma lista de objetos com a seguinte estrutura mínima:

<img width="1430" height="186" alt="image" src="https://github.com/user-attachments/assets/1c3d2906-381a-48e3-b24d-c1c29f31bfaa" />

4. Execução
Execute o script principal:

<img width="1062" height="148" alt="image" src="https://github.com/user-attachments/assets/2608e00f-74ad-4dea-83da-65d3fd21ae82" />

O script criará automaticamente o arquivo dados/dados_limpos.json com o corpus tratado antes de iniciar a indexação e a pesquisa.

📈 Avaliação Qualitativa dos Resultados
O pipeline demonstra um comportamento altamente sofisticado em cenários reais de busca através dos exemplos de teste definidos no código:

Busca por "mudanças na taxa de juros": O motor é capaz de retornar artigos que sequer mencionam a palavra "mudanças" ou "juros", mas abordam termos estritamente relacionados ao ecossistema financeiro brasileiro, como "Copom", "Selic" ou "Política Monetária". Isso prova que o modelo capturou a relação semântica intrínseca que buscas por palavra-chave tradicionais (como TF-IDF) perderiam.

Busca por "mercado de trabalho e desemprego": Graças às regras do TextCleaner que removem lixos estruturais de rodapé (ex: "acesse a pnad contínua completa"), o modelo foca no núcleo da notícia. Ele correlaciona com sucesso conceitos de desemprego a termos como "taxa de desocupação", "geração de vagas" e dados do "IBGE" / "CAGED".

Resiliência a ruídos e preservação de entidades: Ao manter as maiúsculas e corrigir URLs quebradas (ex: g1 . globo . com virando g1.globo.com), o modelo não confunde nomes de portais ou siglas econômicas com palavras comuns do dicionário, elevando a acurácia dos embeddings gerados.

❌ Limitação Qualitativa Conhecida: Por se tratar de um modelo bi-encoder (onde o texto e a query são vetorizados separadamente), ele prioriza o contexto geral. Caso o usuário busque por termos numéricos muito específicos (ex: uma taxa exata de "10,75%"), o modelo pode priorizar o contexto econômico em vez de acertar o número exato. Para contornar isso em produção, recomendaria-se um sistema híbrido (Busca Vetorial + BM25).


