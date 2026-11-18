# SmartAuditFlow Multiagente

## Relatório técnico para a apresentação em grupo

Este documento é material de estudo para as quatro pessoas da equipe. A Seção 5
divide a apresentação em quatro partes técnicas, uma por pessoa, com a mensagem
central de cada uma, o que mostrar, e as perguntas que a banca provavelmente vai
fazer.

Todos os números aqui foram medidos nas execuções de 15 a 18 de novembro de 2026
e podem ser reproduzidos com os comandos do Apêndice A.

---

## 1. Sumário executivo

Partimos do SmartAuditFlow publicado — um pipeline sequencial de auditoria de
contratos inteligentes com LLMs — e fizemos três mudanças: trocamos o pipeline
por uma arquitetura multiagente, removemos o auto-julgamento do modelo, e
implementamos o laço de verificação sugerido pelo professor. Depois medimos
cada uma.

**Resultado 1 — a arquitetura multiagente empata no Top-1 e detecta mais.**
Nos mesmos 134 contratos, com a mesma regra de pontuação aplicada aos dois
lados, a arquitetura multiagente localiza a vulnerabilidade rotulada em 125
contratos contra 117 do baseline. Ela encontra 11 que o baseline não encontra;
o baseline encontra 3 que ela não encontra.

**Resultado 2 — o laço de verificação do professor não ajuda.** Com a
arquitetura e o conjunto de contratos fixos, ligar o laço custa 0,015 de Top-1.
Melhora 15 contratos, piora 20, não muda 100. E ele retira achados 26 vezes
mais do que os refina.

**Resultado 3 — a comparação com o baseline estava enviesada por um bug de
leitura de arquivo.** 41 dos 143 contratos eram descartados silenciosamente da
comparação porque a célula com os achados do baseline continha um caractere
invisível (U+2028). Nesses 41 contratos o baseline pontua 0,268 de Top-1; nos
101 que sobravam, 0,822. O filtro removia quase todas as falhas do baseline.

**O que não conseguimos.** Não superamos o baseline no Top-5 nem no MRR. A
arquitetura reporta cerca de 11 achados por contrato contra os poucos do
baseline, e num benchmark que rotula uma única vulnerabilidade por contrato
isso empurra parte das nossas detecções para fora das cinco primeiras posições.

### Os números de uma olhada

| | valor |
|---|---|
| Contratos no dataset | 143 (142 arquivos; um aparece com duas vulnerabilidades) |
| Auditorias executadas nos dois braços finais | 286 |
| Auditorias válidas | 278 |
| Julgamento cross-provider | 278 de 278 (100%) |
| Auto-validações (modelo julgando a si mesmo) | 0 |
| Testes automatizados | 70, dos quais 69 passam |
| Custo total do projeto em API | US$ 4,27 |

---

## 2. O ponto de partida: o que o artigo faz

O SmartAuditFlow publicado executa, para cada contrato:

- um planejamento inicial da auditoria;
- uma análise, passo a passo, seguindo o plano;
- uma etapa de calibração, em que o modelo avalia a confiança dos próprios
  achados e os ordena.

A terceira etapa é o problema que motivou este trabalho. **O mesmo modelo que
escreveu os achados é quem decide se eles são confiáveis.**

### Por que isso importa

LLMs têm *viés de auto-preferência*: quando avaliam textos, pontuam melhor o
que eles mesmos produziram. Aqui o viés corre na direção perigosa. A nota de
confiança não é um detalhe de relatório — ela é o que ordena os achados, e a
ordenação é exatamente o que as métricas Top-1 e MRR medem. Um sistema que se
auto-valida pode apresentar Top-1 melhor **sem auditar melhor**.

A pergunta de pesquisa do trabalho, então, é: o que acontece com o resultado
quando quem julga não é quem propôs?

---

## 3. A arquitetura proposta

```
START -> context -> plan -> [12 especialistas em paralelo] -> validate
             ^                                                 |    ^
             |                                                 |    |
             |                             nao passou ->    refine
             |                                                 |
             |                                                 v
             +------------------- replan <---------------------+
                                     |
                                     v  (sem pista nova, ou orcamento gasto)
                                 aggregate -> END
```

Sete nós, dois laços aninhados de propósito.

| Nó | Papel |
|---|---|
| `context` | Um briefing compartilhado: propósito do contrato, versão do Solidity, fluxo de valor, modelo de confiança, invariantes. Calculado uma vez, não doze. |
| `plan` | Escreve uma tarefa específica para cada especialista, ancorada no briefing. |
| `specialist` x12 | Uma classe de vulnerabilidade cada, disparados concorrentemente. |
| `validate` | Um painel **independente** tenta refutar cada candidato e só então pontua a confiança. |
| `refine` | Devolve o achado reprovado ao especialista que o propôs, com a objeção anexada. Ele responde com código novo ou retira. |
| `replan` | Decide se vale pagar por uma segunda rodada. |
| `aggregate` | Deduplica, funde corroborações, ordena. |

O **laço interno** (`validate -> refine -> validate`) é o incremento sugerido
pelo professor. O **laço externo** (`replan -> plan`) amplia o plano quando
sobra uma lacuna concreta de cobertura.

### O catálogo de especialistas

Espelha a taxonomia DASP-10 que o conjunto `standardVuln` usa, para que a
cobertura seja auditável e não presumida. Ordenado pela fatia do benchmark:

| Especialista | Contratos no benchmark |
|---|---|
| `unchecked_low_level_calls` | 52 |
| `reentrancy` | 31 |
| `access_control` | 18 |
| `arithmetic` | 15 |
| `bad_randomness` | 8 |
| `denial_of_service` | 6 |
| `time_manipulation` | 5 |
| `front_running` | 4 |
| `data_handling` | 1 |
| `business_logic`, `oracle_pricing`, `gas_efficiency` | para auditorias de protocolo |

---

## 4. Resultados

### 4.1 A regra de pontuação, e por que vocês podem confiar nela

Os números publicados no artigo **não são reproduzíveis** a partir do artefato:
casando o `Issue` da verdade-terreno contra os `Findings` armazenados, só cerca
de 61% dos ranks publicados são reproduzidos, e as divergências são
sistemáticas. Por isso não comparamos contra os números publicados. Em vez
disso, re-pontuamos os achados armazenados do baseline e os nossos com **uma
única regra idêntica e documentada**.

A validação dessa regra é o argumento mais forte da metodologia:

| | n | Top-1 |
|---|---|---|
| Nossa re-pontuação do baseline | 142 | **0,662** |
| Top-1 que o artigo publica, com os rótulos dele | 143 | **0,664** |

Concordância de 0,002, numa regra escrita sem olhar para aqueles rótulos. A
regra é mais rigorosa mais embaixo na lista — o artigo reporta Top-max 1,000
onde a nossa regra dá 0,880 — mas é aplicada de forma idêntica aos dois lados.

### 4.2 Multiagente contra o baseline

Mesmos 134 contratos, mesma regra:

| | Top-1 | Top-5 | Top-max | MRR |
|---|---|---|---|---|
| baseline (artigo, GPT) | 0,672 | **0,873** | 0,873 | **0,755** |
| multiagente, laço desligado | **0,679** | 0,813 | **0,933** | 0,743 |
| multiagente, laço ligado | 0,664 | 0,813 | **0,933** | 0,733 |

Contrato a contrato, o multiagente ganha 24, perde 27, empata 83.

**Detecção: 125 de 134 contra 117 de 134.** Onze vulnerabilidades rotuladas que
só nós encontramos; três que só o baseline encontrou.

Repare numa coisa no baseline: o Top-5 dele (0,873) é exatamente igual ao
Top-max (0,873). Quando ele encontra a vulnerabilidade, ela **já está** entre as
cinco primeiras, sempre. Isso acontece porque ele reporta poucos achados. Nós
reportamos cerca de 11 por contrato, e as nossas detecções a mais caem fora do
top 5. É o mesmo comportamento visto dos dois lados: recall maior, ordenação
pior.

### 4.3 A ablação: o laço do professor

Mesmos 135 contratos, única variável alterada:

| | n | Top-1 | Top-5 | Top-max | MRR |
|---|---|---|---|---|---|
| laço DESLIGADO | 135 | **0,674** | 0,807 | 0,933 | **0,738** |
| laço LIGADO | 135 | 0,659 | 0,807 | 0,933 | 0,728 |
| diferença | | **-0,015** | 0,000 | 0,000 | **-0,010** |

Contrato a contrato: melhora 15, piora 20, não muda 100.

Restringindo aos 131 contratos em que o laço **efetivamente agiu**, o quadro é
o mesmo (-0,015 de Top-1). Não é diluição da amostra: o mecanismo foi acionado
e não produziu ganho.

**O número que explica o porquê: 1.297 achados retirados contra 49 refinados.**
O laço faz o especialista desistir 26 vezes mais do que defender a alegação com
evidência nova.

Isso não é o comportamento que o desenho pretendia. A intenção era "reprovou,
volte e justifique melhor". Na prática virou "reprovou, retire". As travas
contra reescrever-até-alguém-aceitar funcionaram, e talvez bem demais: exigir
evidência nova, conferida contra o código-fonte, é uma barra que o especialista
raramente transpõe, então ele retira.

### 4.4 O viés que quase passou despercebido

Esta é a correção com maior efeito em todo o projeto, e vale contar como
descoberta, não só como resultado.

O comparador descartava 42 das 143 linhas do baseline como "ilegíveis". Isso
parecia uma propriedade ruim dos dados. Não era: 41 daquelas células são reprs
Python completos e bem-formados que contêm o caractere invisível **U+2028 LINE
SEPARATOR** entre os elementos. O `ast.literal_eval` rejeita esse caractere como
não-imprimível no código-fonte, e a célula inteira ia fora.

O descarte **não era neutro**:

| baseline, re-pontuado | n | Top-1 | Top-5 | Top-max | MRR |
|---|---|---|---|---|---|
| células que sempre liam | 101 | 0,822 | 0,970 | 0,970 | 0,885 |
| células recuperadas pela correção | 41 | **0,268** | 0,634 | 0,659 | 0,415 |
| todas | 142 | **0,662** | 0,873 | 0,880 | 0,749 |

As 41 células descartadas eram quase todas casos em que o baseline **falhou**.
Filtrá-las apresentava o baseline em 0,822 em vez de 0,662 — e foi contra esse
0,822 que medimos o nosso trabalho por três dias.

A lição metodológica, que vale dizer na apresentação: **um filtro de dados nunca
é neutro até você medir se ele é.** Bastou comparar a taxa de acerto dentro e
fora do filtro para ver o problema.

---

## 5. Divisão da apresentação

Quatro partes. Cada uma tem uma mensagem que cabe numa frase, o material a
mostrar, e as perguntas que a pessoa precisa saber responder sem consultar nada.

### 5.1 Pessoa 1 — A arquitetura multiagente

**Mensagem em uma frase:** trocamos um pipeline sequencial por um grafo com
orquestrador, doze especialistas em paralelo e um validador adversarial.

**O que mostrar**

- O diagrama `docs/arquitetura.drawio` — ele tem as duas arquiteturas
  empilhadas, a do artigo em cima e a nossa embaixo, com os três incrementos
  marcados.
- A tabela de nós (Seção 3) e o catálogo DASP-10.

**Onde está no código:** `src/agents/graph.py`, `src/agents/nodes.py`,
`src/agents/state.py`.

**Perguntas que você precisa saber responder**

*Por que um grafo e não só chamadas sequenciais?* Porque os doze especialistas
são independentes entre si e podem rodar ao mesmo tempo; e porque os dois laços
(refinar, replanejar) precisam de arestas condicionais, que num encadeamento
linear viram `if` aninhado e param de ser auditáveis.

*Por que doze especialistas, e por que esses?* Espelham a taxonomia DASP-10 que
o próprio conjunto de testes usa. Assim a cobertura é verificável: dá para
apontar qual especialista cobre cada classe do benchmark, em vez de afirmar que
"o modelo olha tudo".

*Por que calcular o contexto uma vez só?* Porque os doze especialistas
precisariam da mesma leitura básica do contrato. Calcular uma vez e distribuir
economiza onze chamadas por contrato e, mais importante, garante que todos
partem da mesma interpretação do código.

*Qual a diferença entre os dois laços?* O interno conserta um achado que já
existe. O externo pede achados novos. O interno é barato e é o que mexe no
Top-1; o externo é caro e só dispara se sobrar uma lacuna concreta.

### 5.2 Pessoa 2 — Julgamento independente

**Mensagem em uma frase:** o artigo deixa o mesmo modelo julgar o que ele mesmo
escreveu, e é dessa nota que o ranking inteiro depende.

**O que mostrar**

- Os níveis de independência e a regra de desempate.
- Os números: 278 auditorias válidas, 100% com julgamento cross-provider, zero
  auto-validações.

**Onde está no código:** `src/agents/validator.py`, `src/agents/llm.py` (função
`select_judges`).

**O conteúdo**

Os juízes são escolhidos **por achado**, contra quem o propôs:

| Nível | Significado |
|---|---|
| `cross-provider` | Juiz de outra família de provedor. É o alvo. |
| `cross-model` | Mesma família, outro checkpoint. Fraco: linhagem compartilhada. |
| `self` | O proponente julgou a si mesmo. Enviesado, e sinalizado. |
| `unjudged` | Nenhum juiz respondeu. O resultado é inválido e sai da tabela. |

A independência é classificada no nível da **família de provedor**, não do id do
modelo: `gpt-4o` julgando `gpt-4o-mini` compartilha dados de treino e
alinhamento, então o efeito de auto-preferência não desaparece só porque o
checkpoint mudou.

Com painel de mais de um juiz, eles rodam em paralelo e nunca veem a resposta um
do outro, para que nenhum seja ancorado por um veredito anterior. Empate resolve
para `uncertain`, **nunca** para `rejected`.

**Perguntas que você precisa saber responder**

*Por que família e não id do modelo?* Porque o viés de auto-preferência vem de
dados de treino e alinhamento compartilhados, não do número da versão.
Classificar por id deixaria `gpt-4o` julgando `gpt-4o-mini` passar como
independente, o que é mentira.

*Por que empate vira "incerto" e não "rejeitado"?* Rejeição joga o achado para o
fim do relatório. Resolver um empate como rejeição enterraria uma
vulnerabilidade possivelmente real e custaria exatamente o Top-1 e o MRR que o
desenho persegue. Rejeição tem que ser conquistada por maioria.

*O que impede o sistema de reescrever o achado até algum juiz aceitar?* Três
travas: o número de revisões é limitado; a revisão tem que citar código que
ainda não tinha citado, e essa citação é conferida contra o fonte (linha
inventada é descartada); e a revisão volta **para o mesmo juiz que objetou**,
nunca para um juiz novo.

*E quando dois especialistas concordam?* Dois modelos concordando só é evidência
forte se eles não forem o mesmo modelo. Concordância dentro da mesma família
compartilha pontos cegos, então ela vale um terço do crédito que vale uma
concordância entre provedores diferentes.

### 5.3 Pessoa 3 — Confiabilidade da execução

**Mensagem em uma frase:** um número de benchmark só vale se a execução não
inventou nada, e três bugs quase produziram tabelas convincentes e falsas.

**O que mostrar**

- Os três tipos de erro 429 e por que confundi-los custa caro.
- O caso do juiz morto (abaixo). É a melhor história técnica do projeto.

**Onde está no código:** `src/agents/llm.py`, `src/agents/providers.py`.

**Os três bugs**

*Erros de conexão descartando achados em silêncio.* A lista de erros que
justificam nova tentativa não incluía falhas de socket. Uma queda de TCP durante
o disparo paralelo descartava o achado daquele especialista — e o resultado
parecia "o modelo não achou nada", não "houve um erro". Foram 48 ocorrências
numa única execução. **Essa é a classe mais perigosa de falha: a que se parece
com um resultado.**

*Juízes mortos contados como tendo julgado.* O validador contava toda tentativa
de julgamento, inclusive as que falharam. Uma execução de 143 contratos reportou
`cross-provider`, zero erros e zero auto-validações **enquanto 1.869 chamadas de
juiz tinham falhado**. A tabela era convincente e inteiramente falsa. A correção
separa "juízes que responderam" de "juízes acionados", introduz o nível
`unjudged`, e marca a execução como inválida. Foi essa trava que, depois, pegou
mais duas execuções ruins antes de virarem número publicado.

*Cota diária tratada como limite de taxa.* Um HTTP 429 pode significar três
coisas diferentes:

| Tipo | Exemplo | O que fazer |
|---|---|---|
| taxa por minuto | "rate limit, retry in 20s" | esperar e repetir |
| cota diária | "free_tier_requests, retry in 7h" | parar: repetir não adianta |
| limite de gasto | "organization_spend_limit_exceeded" | parar: não é temporário |

Tratar os três como o primeiro fez o sistema repetir, por horas, chamadas que
nunca iam passar.

**O limitador adaptativo.** Os limites anunciados pelos provedores não batiam
com a realidade — a NVIDIA anuncia 40 req/min; medimos que 77/min sequenciais
passam, mas que volume sustentado não. Em vez de chutar um número, o cliente usa
um esquema AIMD, o mesmo do controle de congestionamento do TCP: a cada recusa
por taxa ele corta a vazão pela metade; depois de uma sequência de sucessos,
sobe 25% até o teto declarado. Ele **aprende** o limite real em vez de assumi-lo.

**Perguntas que você precisa saber responder**

*Como vocês distinguem um 429 de cota de um 429 de taxa?* Pelo texto da resposta
e pelo tempo de espera sugerido. Se a mensagem cita cota diária ou pede para
voltar em horas, é permanente para aquele dia. Isso está codificado e testado.

*Qual foi o custo?* US$ 4,27 em todo o projeto, com 9.319 respostas distintas em
cache. O cache guarda a resposta por hash do prompt, então repetir uma análise
idêntica não gera cobrança nova.

*Como vocês sabem que o código funciona sem gastar API?* 70 testes automatizados
que não tocam a rede; 69 passam. O único que falha é um teste de velocidade
sensível à carga da máquina, não de correção.

### 5.4 Pessoa 4 — Avaliação e resultados

**Mensagem em uma frase:** duas correções na régua de medir mudaram a conclusão
do trabalho, e a ablação mostra que o incremento proposto não ajuda.

**O que mostrar**

- As tabelas das Seções 4.2, 4.3 e 4.4.
- A validação da regra: 0,662 contra 0,664.

**Onde está no código:** `benchmark/scoring.py`, `benchmark/run.py`.

**As métricas, em uma linha cada**

| Métrica | O que mede |
|---|---|
| Top-1 | A vulnerabilidade rotulada é o primeiro achado do relatório? |
| Top-5 | Ela está entre os cinco primeiros? |
| Top-max | Ela está em algum lugar do relatório? (é detecção, não ordenação) |
| MRR | 1/posição, média. Premia estar perto do topo. |

**As duas correções**

*Correção 1 — qual candidato conta.* A implementação original contradizia a
própria documentação: o texto prometia "o rank do **primeiro** achado que
descreve a vulnerabilidade", mas o código pegava o candidato com maior
sobreposição de palavras. Para uma métrica de ordenação a documentação estava
certa: quem lê um relatório de cima para baixo encontra o problema na primeira
entrada que o descreve. A regra antiga **cobrava do relatório por descrever o
mesmo problema duas vezes**. Corrigida, e aplicada identicamente aos dois lados,
**os dois lados melhoraram** — e é isso que torna a correção defensável. Se só o
nosso lado tivesse melhorado, seria indistinguível de ajustar a métrica ao
resultado.

*Correção 2 — quais contratos entram.* O caso U+2028 da Seção 4.4.

**Perguntas que você precisa saber responder**

*Como vocês sabem que a regra de vocês não favorece o lado de vocês?* Três
argumentos, nesta ordem: (a) ela reproduz o Top-1 que o artigo publica com os
rótulos dele, 0,662 contra 0,664, e foi escrita sem olhar para eles; (b) as duas
correções foram aplicadas identicamente aos dois lados, e na primeira os dois
melhoraram; (c) a regra é mais rigorosa que a do artigo mais embaixo na lista, o
que trabalha **contra** nós, não a favor.

*Por que não ajustaram os pesos do ranking para melhorar o Top-1?* Porque isso
seria ajustar ao conjunto de teste. Nove variantes de peso foram medidas e todas
caem entre 0,257 e 0,310 de Top-1 — ou seja, o peso não é o gargalo. Nenhuma
dessas escolhas está no código entregue.

*Então por que vocês perdem no Top-5 se detectam mais?* Porque reportamos cerca
de 11 achados por contrato e o benchmark rotula **um**. Outros achados genuínos
de severidade alta, ordenados acima do rotulado, contam como falha. Isso é uma
propriedade do protocolo de avaliação, não da auditoria.

*Por que o laço do professor falhou?* Porque ele converteu objeção em retirada,
não em refinamento: 1.297 retiradas contra 49 refinamentos. A exigência de
evidência nova conferida contra o código é severa demais; o especialista acha
mais fácil desistir do que atender.

---

## 6. Limitações e trabalho futuro

**O que não está provado.** A diferença de 0,007 no Top-1 a nosso favor, e a de
-0,015 na ablação, são pequenas demais para serem distinguidas de ruído com 134
contratos. O que está solidamente estabelecido é a diferença de detecção (125
contra 117) e a razão retirada/refinamento (26 para 1).

**Uma única execução por braço.** Não medimos variância entre execuções. Com
temperatura não-zero, repetir cada braço três vezes seria o próximo passo óbvio
e barato.

**O benchmark rotula uma vulnerabilidade por contrato.** Isso penaliza
sistematicamente qualquer sistema de recall alto. Uma avaliação mais justa
exigiria verdade-terreno com múltiplos rótulos por contrato, que o conjunto não
tem.

**Dois caminhos para transformar isto em ganho.** Afrouxar a exigência de
evidência do refinador, que hoje empurra para retirada; ou atacar a geração de
candidatos, já que os 11 achados por contrato são o que custa o Top-1.

**O incremento 2 do professor (ablação de prompts) não foi feito.** O diretório
`promptOptimization/` do repositório original não está conectado ao código em
`src/`: os prompts são literais no próprio `nodes.py`. Conectar os dois é um
trabalho à parte.

---

## Apêndice A — Como reproduzir

```
# Testes, sem chaves de API e sem rede
.venv/Scripts/python.exe tests/test_providers.py
.venv/Scripts/python.exe tests/test_judge_independence.py
.venv/Scripts/python.exe tests/test_refinement_loop.py
.venv/Scripts/python.exe tests/test_agents_graph.py
.venv/Scripts/python.exe tests/test_agents_engine.py

# Re-pontuar os baselines publicados (sem custo, sem chamadas de API)
.venv/Scripts/python.exe -m benchmark.run --rescore-only

# O que esta configurado, e cada chave realmente responde?
.venv/Scripts/python.exe check_providers.py --probe

# Os dois bracos da ablacao.
# --judge fixa o conjunto de juizes; --judge-top-n 8 limita o julgamento aos 8
# achados mais confiantes, porque so a cabeca do ranking pode mexer nas metricas.
.venv/Scripts/python.exe -m benchmark.run --model openai \
    --judge nvidia --judge gemini --judge-top-n 8 \
    --max-refine-rounds 1 --workers 3 --out armA.csv    # laco ligado

.venv/Scripts/python.exe -m benchmark.run --model openai \
    --judge nvidia --judge gemini --judge-top-n 8 \
    --max-refine-rounds 0 --workers 3 --out armB.csv    # laco desligado
```

## Apêndice B — Mapa do código

| Caminho | O que é |
|---|---|
| `src/agents/graph.py` | Montagem do grafo, arestas condicionais, disparo paralelo |
| `src/agents/nodes.py` | Os sete nós e os prompts de cada agente |
| `src/agents/validator.py` | Painel de juízes, veredito, pontuação de ranking |
| `src/agents/llm.py` | Cliente, seleção de juízes, limitador AIMD, cache, repetições |
| `src/agents/providers.py` | Registro de provedores, famílias, limites medidos |
| `benchmark/scoring.py` | A regra de casamento e as métricas |
| `benchmark/run.py` | Execução do benchmark, re-pontuação, custo |
| `tests/` | 70 testes, nenhum toca a rede |
| `docs/arquitetura.drawio` | Diagrama das duas arquiteturas e dos incrementos |
| `ARCHITECTURE.md` | Documentação técnica completa, com as decisões e os porquês |
