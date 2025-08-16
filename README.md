Este projeto utiliza um modelo agroclimático para executar simulações de Monte Carlo, permitindo que agrônomos, produtores rurais e pesquisadores identifiquem as datas de plantio mais estratégicas. A ferramenta analisa o trade-off entre o potencial de desenvolvimento da cultura (medido em GDU) e os riscos associados à disponibilidade de água (dias de estresse hídrico).

O modelo foi parametrizado com dados de referência para a região do médio-norte de Mato Grosso, Brasil, mas pode ser adaptado para outras realidades.

## ✨ Funcionalidades

-   **Painel Interativo:** Ajuste facilmente os parâmetros da simulação, como o período de análise, intervalo entre as janelas, duração do ciclo da cultura e número de iterações do Monte Carlo.
-   **Simulação Dinâmica:** Execute complexas simulações agroclimáticas diretamente no navegador com um único clique.
-   **Visualização de Dados:** Visualize os resultados consolidados em uma tabela e em um gráfico interativo que compara o GDU acumulado com os dias de estresse hídrico.
-   **Análise Automatizada:** Receba recomendações e insights automáticos sobre as melhores janelas de plantio com base nos resultados da simulação.

## ⚙️ Como Funciona

O núcleo da aplicação é uma função que simula o desenvolvimento diário de uma cultura. O processo considera:

1.  **Geração de Clima Estocástico:** As condições diárias de chuva e nebulosidade são geradas usando Cadeias de Markov, e a intensidade da precipitação segue uma distribuição Gama.
2.  **Balanço Hídrico do Solo:** O modelo calcula a disponibilidade de água no solo em duas camadas, considerando precipitação, evapotranspiração (calculada pelo método de Penman-Monteith) e runoff.
3.  **Cálculo de Graus-Dia (GDU):** O acúmulo de GDU é calculado diariamente, mas ajustado por um fator de estresse hídrico. Se a disponibilidade de água no solo cai abaixo de um limiar crítico, o desenvolvimento da planta (acúmulo de GDU) é penalizado.
4.  **Análise de Monte Carlo:** Para cada data de plantio, o ciclo completo da cultura é simulado centenas de vezes, gerando uma distribuição de resultados prováveis. Os valores medianos são então usados para a análise final.

## 🚀 Tecnologias Utilizadas

-   **Backend & Frontend:** [Python](https://www.python.org/) com [Streamlit](https://streamlit.io/)
-   **Análise de Dados:** [Pandas](https://pandas.pydata.org/) e [NumPy](https://numpy.org/)
-   **Cálculos Científicos:** [SciPy](https://scipy.org/)
-   **Visualização de Dados:** [Matplotlib](https://matplotlib.org/) e [Seaborn](https://seaborn.pydata.org/)

## 💻 Como Executar Localmente

Para executar este projeto em sua máquina local, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/anielloheronville/Dashboard-GDU.git](https://github.com/anielloheronville/Dashboard-GDU.git)
    cd Dashboard-GDU
    ```

2.  **(Recomendado) Crie um ambiente virtual:**
    ```bash
    python -m venv venv
    # No Windows
    venv\Scripts\activate
    # No macOS/Linux
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    O arquivo `requirements.txt` contém todas as bibliotecas necessárias.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute a aplicação Streamlit:**
    ```bash
    streamlit run app.py
    ```

5.  Abra seu navegador e acesse `http://localhost:8501`.

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](URL_DO_SEU_ARQUIVO_DE_LICENCA) para mais detalhes.

---

*Desenvolvido por Aniello Heronville.*
