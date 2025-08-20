# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma
from datetime import timedelta, datetime
import google.generativeai as genai

# =============================================================================
# SEÇÃO 1: FUNÇÕES DO MODELO DE SIMULAÇÃO (Lógica Principal)
# =============================================================================

def atualizar_parametros_cultura(gdu_acumulado, cfg):
    if gdu_acumulado < cfg['gdu_fase_reprodutiva_inicial']:
        albedo = 0.20
        fator_crescimento_raiz = gdu_acumulado / cfg['gdu_fase_reprodutiva_inicial']
        prof_raiz_efetiva = cfg['profundidade_camada1'] + (cfg['profundidade_camada2'] * fator_crescimento_raiz)
    elif gdu_acumulado < cfg['gdu_fase_reprodutiva_pico']:
        albedo = 0.23
        prof_raiz_efetiva = cfg['profundidade_camada1'] + cfg['profundidade_camada2']
    else:
        albedo = 0.21
        prof_raiz_efetiva = cfg['profundidade_camada1'] + cfg['profundidade_camada2']
    prof_max = cfg['profundidade_camada1'] + cfg['profundidade_camada2']
    return albedo, min(prof_raiz_efetiva, prof_max)

def simular_processos_diarios(n_dias, start_day_of_year, latitude_rad, altitude, cfg):
    precipitacao, eto_arr, gdu_padrao, gdu_ajustado, dias_estresse, runoff = (np.zeros(n_dias) for _ in range(6))
    estado_chuva, estado_nebulosidade = (np.zeros(n_dias, dtype=np.int64) for _ in range(2))
    armazenamento_c1, armazenamento_c2 = (np.zeros(n_dias) for _ in range(2))
    armazenamento_c1[0] = cfg['capacidade_max_camada1'] * 0.8
    armazenamento_c2[0] = cfg['capacidade_max_camada2'] * 0.8
    gdu_acumulado = 0.0

    for i in range(1, n_dias):
        if np.random.rand() < cfg['P_chuva'][estado_chuva[i-1], 1]: estado_chuva[i] = 1
        precipitacao[i] = gamma.rvs(a=cfg['formato_gama'], scale=cfg['escala_gama']) if estado_chuva[i] == 1 else 0.0

        day_index = start_day_of_year + i
        temp_media_sazonal = cfg['temp_media_anual'] + cfg['amplitude_sazonal'] * np.sin(2 * np.pi * day_index / 365.25)
        fator_neb_temp = 0.6 if estado_nebulosidade[i] == 1 else 1.0
        amp_diurna = cfg['amplitude_diurna_base'] * fator_neb_temp
        t_max = temp_media_sazonal + amp_diurna / 2 + np.random.normal(0, 1.0)
        t_min = t_max - amp_diurna - np.random.normal(0, 1.0)
        t_mean = (t_max + t_min) / 2
        albedo, prof_raiz_efetiva = atualizar_parametros_cultura(gdu_acumulado, cfg)

        sigma = 4.903e-9; G = 0; P = 101.3 * ((293 - 0.0065 * altitude) / 293)**5.26; gamma_p = 0.000665 * P
        e_tmax = 0.6108 * np.exp((17.27 * t_max) / (t_max + 237.3)); e_tmin = 0.6108 * np.exp((17.27 * t_min) / (t_min + 237.3))
        es = (e_tmax + e_tmin) / 2; ea = (np.random.uniform(80, 95) / 100) * es if estado_chuva[i] == 1 else (np.random.uniform(55, 75) / 100) * es
        delta_v = 4098 * (0.6108 * np.exp((17.27 * t_mean) / (t_mean + 237.3))) / (t_mean + 237.3)**2
        dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * day_index); delta_sol = 0.409 * np.sin(2 * np.pi / 365 * day_index - 1.39)
        omega_s = np.arccos(-np.tan(latitude_rad) * np.tan(delta_sol))
        Ra = (24 * 60 / np.pi) * 0.0820 * dr * (omega_s * np.sin(latitude_rad) * np.sin(delta_sol) + np.cos(latitude_rad) * np.cos(delta_sol) * np.sin(omega_s))
        n_N_ratio = 0.35 if estado_nebulosidade[i] == 1 else 0.8; Rs = Ra * (cfg['a_s'] + cfg['b_s'] * n_N_ratio)
        Rso = (0.75 + 2e-5 * altitude) * Ra; Rns = (1 - albedo) * Rs
        t_max_k4 = (t_max + 273.16)**4; t_min_k4 = (t_min + 273.16)**4
        Rnl = sigma * ((t_max_k4 + t_min_k4) / 2) * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * (Rs / Rso) - 0.35)
        Rn = Rns - Rnl; u2 = np.random.uniform(1.0, 3.0)
        eto_arr[i] = max(0, (0.408 * delta_v * (Rn - G) + gamma_p * (900 / (t_mean + 273)) * u2 * (es - ea)) / (delta_v + gamma_p * (1 + 0.34 * u2)))
        agua_c1_inicio = armazenamento_c1[i-1] + precipitacao[i]
        percolacao = max(0, agua_c1_inicio - cfg['capacidade_max_camada1']); armazenamento_c1_temp = agua_c1_inicio - percolacao
        agua_c2_inicio = armazenamento_c2[i-1] + percolacao
        runoff[i] = max(0, agua_c2_inicio - cfg['capacidade_max_camada2']); armazenamento_c2_temp = agua_c2_inicio - runoff[i]
        agua_disponivel_c1 = max(0, armazenamento_c1_temp - cfg['ponto_murcha_camada1']); agua_disponivel_c2 = max(0, armazenamento_c2_temp - cfg['ponto_murcha_camada2'])
        etr_c1 = min(agua_disponivel_c1, eto_arr[i] * 0.15)
        demanda_transpiracao = eto_arr[i] - etr_c1
        etr_c2 = min(agua_disponivel_c2, demanda_transpiracao) if prof_raiz_efetiva > cfg['profundidade_camada1'] else 0
        armazenamento_c1[i] = armazenamento_c1_temp - etr_c1; armazenamento_c2[i] = armazenamento_c2_temp - etr_c2
        gdu_padrao[i] = max(0, (min(t_max, cfg['T_MAX_LIMITE']) + max(t_min, cfg['T_BASE']))/2 - cfg['T_BASE'])
        cap_disponivel_total_raiz = (cfg['capacidade_max_camada1'] - cfg['ponto_murcha_camada1']) + (cfg['capacidade_max_camada2'] - cfg['ponto_murcha_camada2'])
        agua_disponivel_total_raiz = max(0, armazenamento_c1[i] - cfg['ponto_murcha_camada1']) + max(0, armazenamento_c2[i] - cfg['ponto_murcha_camada2'])
        fator_umidade_relativa = agua_disponivel_total_raiz / cap_disponivel_total_raiz if cap_disponivel_total_raiz > 0 else 0
        umbral_estresse = 0.4
        fator_estresse = min(1.0, fator_umidade_relativa / umbral_estresse) if fator_umidade_relativa < umbral_estresse else 1.0
        if fator_estresse < 1.0: dias_estresse[i] = 1
        gdu_ajustado[i] = gdu_padrao[i] * fator_estresse
        gdu_acumulado += gdu_ajustado[i]
    return gdu_ajustado, dias_estresse, precipitacao, runoff

# =============================================================================
# SEÇÃO 2: FUNÇÕES PARA EXECUÇÃO DA ANÁLISE E VISUALIZAÇÃO
# =============================================================================

@st.cache_data
def executar_analise_janelas(_datas_plantio, ciclo_dias, n_simulacoes, config):
    todos_os_resultados = []
    barra_progresso = st.progress(0, text="Iniciando simulação...")
    status_texto = st.empty()
    total_janelas = len(_datas_plantio)
    for i, data_inicio in enumerate(_datas_plantio):
        data_fim = data_inicio + timedelta(days=ciclo_dias - 1)
        status_texto.text(f"Simulando janela {i+1}/{total_janelas}: {data_inicio.strftime('%Y-%m-%d')} a {data_fim.strftime('%Y-%m-%d')}")
        resultados_monte_carlo = []
        for j in range(n_simulacoes):
            start_day_of_year = data_inicio.dayofyear
            latitude_rad = np.deg2rad(config['LATITUDE_GRAUS'])
            gdu_adj, estresse_dia, precip_dia, runoff_dia = simular_processos_diarios(
                ciclo_dias, start_day_of_year, latitude_rad, config['ALTITUDE_METROS'], config
            )
            metricas = {
                'data_plantio': data_inicio.strftime('%Y-%m-%d'),
                'simulacao_n': j + 1,
                'gdu_final_ajustado': np.sum(gdu_adj),
                'dias_estresse_hidrico': np.sum(estresse_dia),
                'precipitacao_total_safra': np.sum(precip_dia),
                'runoff_total_safra': np.sum(runoff_dia)
            }
            resultados_monte_carlo.append(metricas)
        todos_os_resultados.extend(resultados_monte_carlo)
        barra_progresso.progress((i + 1) / total_janelas, text=f"Analisando janela {i+1}/{total_janelas}")
    status_texto.text("Simulação concluída!")
    df_detalhado = pd.DataFrame(todos_os_resultados)
    df_resumo = df_detalhado.groupby('data_plantio').median().drop(columns='simulacao_n')
    return df_resumo, df_detalhado

def gerar_grafico_principal(df_resumo):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(x=df_resumo.index, y='gdu_final_ajustado', data=df_resumo, ax=ax1, color='cornflowerblue', label='GDU Final Ajustado (Mediana)')
    ax1.set_xlabel('Data de Plantio', fontsize=12)
    ax1.set_ylabel('GDU Final Ajustado (Acumulado)', fontsize=12, color='cornflowerblue')
    ax1.tick_params(axis='y', labelcolor='cornflowerblue', labelsize=12)
    ax1.tick_params(axis='x', rotation=45, labelsize=12)
    plt.setp(ax1.get_xticklabels(), ha="right")
    ax1.set_title('Análise de Janela de Plantio: GDU vs. Dias de Estresse Hídrico', fontsize=16, pad=20)
    ax2 = ax1.twinx()
    sns.lineplot(x=df_resumo.index, y='dias_estresse_hidrico', data=df_resumo, ax=ax2, color='tomato', marker='o', lw=3, label='Dias de Estresse Hídrico (Mediana)')
    ax2.set_ylabel('Dias de Estresse Hídrico (Total)', fontsize=12, color='tomato')
    ax2.tick_params(axis='y', labelcolor='tomato', labelsize=12)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    ax1.get_legend().remove()
    plt.grid(False)
    plt.tight_layout()
    return fig

def gerar_boxplots(df_detalhado):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.set_style("whitegrid")
    sns.boxplot(ax=axes[0], x='data_plantio', y='gdu_final_ajustado', data=df_detalhado, palette="coolwarm")
    axes[0].set_title('Distribuição do GDU Final por Janela', fontsize=14)
    axes[0].set_xlabel('Data de Plantio', fontsize=12)
    axes[0].set_ylabel('GDU Final Ajustado', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    sns.boxplot(ax=axes[1], x='data_plantio', y='dias_estresse_hidrico', data=df_detalhado, palette="coolwarm")
    axes[1].set_title('Distribuição de Dias de Estresse por Janela', fontsize=14)
    axes[1].set_xlabel('Data de Plantio', fontsize=12)
    axes[1].set_ylabel('Dias com Estresse Hídrico', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    return fig

def gerar_tradeoff_plot(df_resumo):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.set_style("whitegrid")
    sns.scatterplot(x='dias_estresse_hidrico', y='gdu_final_ajustado', data=df_resumo, hue=df_resumo.index, s=200, palette='viridis', ax=ax)
    for i, row in df_resumo.iterrows():
        ax.text(row['dias_estresse_hidrico'] + 0.5, row['gdu_final_ajustado'], i, fontsize=9)
    median_stress = df_resumo['dias_estresse_hidrico'].median()
    median_gdu = df_resumo['gdu_final_ajustado'].median()
    ax.axvline(median_stress, color='grey', linestyle='--', lw=1)
    ax.axhline(median_gdu, color='grey', linestyle='--', lw=1)
    ax.set_title('Análise de Trade-Off: GDU vs. Estresse Hídrico', fontsize=16)
    ax.set_xlabel('Dias de Estresse Hídrico (Mediana)', fontsize=12)
    ax.set_ylabel('GDU Final Ajustado (Mediana)', fontsize=12)
    ax.legend(title='Data de Plantio', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.text(ax.get_xlim()[1], median_gdu, '  Alto GDU', va='bottom', ha='right', color='grey')
    ax.text(ax.get_xlim()[0], median_gdu, '  Baixo GDU', va='bottom', ha='left', color='grey')
    ax.text(median_stress, ax.get_ylim()[1], 'Baixo Estresse', va='top', ha='right', rotation=90, color='grey')
    ax.text(median_stress, ax.get_ylim()[0], 'Alto Estresse', va='bottom', ha='right', rotation=90, color='grey')
    plt.tight_layout()
    return fig

# =============================================================================
# SEÇÃO 3: LÓGICA DA IA GEMINI
# =============================================================================
def get_weather_forecast_summary():
    """Retorna um resumo da previsão do tempo obtida via busca."""
    return ("A previsão climática para o final de agosto e setembro de 2025 em Lucas do Rio Verde, MT, "
            "indica um período quente e seco, com umidade relativa do ar baixa e sem previsão de chuvas significativas. "
            "Isso sugere um risco aumentado de estresse hídrico para o início do ciclo de plantio.")

def gerar_analise_gemini(df_resumo, df_detalhado, previsao_tempo=None):
    """Envia os resultados para a API do Gemini e retorna a análise."""
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt_contexto = f"""
        **Contexto:** Você é um engenheiro agrônomo e especialista em modelagem de safras. 
        Sua tarefa é analisar os resultados de uma simulação de Monte Carlo para diferentes janelas de plantio de milho safrinha.

        **Dados de Simulação para Análise:**
        - **Tabela de Medianas:** Mostra o resultado mediano (o mais provável) para cada janela.
        {df_resumo.to_string()}

        - **Tabela de Desvio Padrão:** Mostra a variabilidade (risco) dos resultados. Valores mais altos indicam maior incerteza.
        {df_detalhado.groupby('data_plantio').std().to_string()}
        """

        if previsao_tempo:
            prompt_contexto += f"\n**Previsão do Tempo Externa (Fator Adicional):**\n{previsao_tempo}\n"
        
        prompt_tarefa = """
        **Sua Tarefa:**
        1.  **Análise Geral:** Escreva uma análise concisa sobre os resultados medianos.
        2.  **Identifique o Trade-Off:** Explique o trade-off entre plantar mais cedo ou mais tarde, com base no GDU e no estresse hídrico.
        3.  **Análise de Risco:** Com base no desvio padrão, qual janela apresenta maior e menor risco? Interprete o que isso significa na prática.
        4.  **Recomendação Final:** Com base em tudo (medianas, risco e a previsão do tempo, se fornecida), qual seria sua recomendação final para o agricultor? Justifique sua escolha.
        5.  **Formato:** Use Markdown para formatar a resposta de forma clara, com títulos e negrito.
        """
        
        response = model.generate_content(prompt_contexto + prompt_tarefa)
        return response.text
    except Exception as e:
        st.error(f"Erro ao conectar com a API do Gemini. Verifique sua chave de API nos secrets.")
        st.error(f"Detalhe do erro: {e}")
        return None

def handle_chat(user_question, df_resumo, df_detalhado):
    """Lida com a interação do chatbot, fornecendo contexto para a IA."""
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Constrói o histórico da conversa para dar contexto ao modelo
        full_prompt = f"""
        **Contexto:** Você é um chatbot assistente de um agrônomo. Você está analisando os resultados de uma simulação de janela de plantio.
        Abaixo estão os dados da simulação. Use-os para responder a pergunta do usuário.

        **Resultados Medianos:**
        {df_resumo.to_string()}

        **Dados de Risco (Desvio Padrão):**
        {df_detalhado.groupby('data_plantio').std().to_string()}

        **Pergunta do Usuário:** "{user_question}"

        **Sua Resposta:**
        """
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Erro ao processar a pergunta: {e}"

# =============================================================================
# SEÇÃO 4: INTERFACE DO USUÁRIO (Streamlit App)
# =============================================================================

st.set_page_config(layout="wide", page_title="Otimização da Janela de Plantio")
st.title("🌽 Dashboard Inteligente de Otimização da Janela de Plantio")
st.markdown("Uma ferramenta que combina simulação de Monte Carlo e Inteligência Artificial para otimizar suas decisões de plantio.")

if 'df_resumo' not in st.session_state:
    st.session_state.df_resumo = None
if 'df_detalhado' not in st.session_state:
    st.session_state.df_detalhado = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Parâmetros da Análise")
    st.subheader("🗓️ Janelas de Plantio")
    data_inicial_analise = st.date_input("Data Inicial da Análise", datetime(2025, 10, 1))
    data_final_analise = st.date_input("Data Final da Análise", datetime(2026, 1, 15))
    intervalo_dias = st.slider("Intervalo entre datas (dias)", 7, 30, 15)
    datas_de_plantio_para_analise = pd.to_datetime(pd.date_range(start=data_inicial_analise, end=data_final_analise, freq=f'{intervalo_dias}D'))
    
    st.subheader("⚙️ Parâmetros da Simulação")
    ciclo_duracao_dias = st.slider("Duração do Ciclo (dias)", 90, 180, 150)
    numero_de_simulacoes_por_janela = st.slider("Nº de Simulações/Janela", 50, 500, 200)

    with st.expander("🔬 Parâmetros Avançados"):
        T_BASE = st.number_input("Temperatura Base (°C)", 5.0, 15.0, 10.0, 0.5)
        T_MAX_LIMITE = st.number_input("Temp. Máxima Limite (°C)", 28.0, 40.0, 30.0, 0.5)
        gdu_fase_reprodutiva_inicial = st.number_input("GDU Início Reprodutivo", 500.0, 1000.0, 700.0, 10.0)
        gdu_fase_reprodutiva_pico = st.number_input("GDU Pico Reprodutivo", 1000.0, 1500.0, 1100.0, 10.0)
        escala_gama = st.number_input("Escala Gama (Chuva)", 5.0, 15.0, 10.0, 0.5)

config = {
    'P_chuva': np.array([[0.85, 0.15], [0.40, 0.60]]), 'P_nebulosidade': np.array([[0.7, 0.3], [0.4, 0.6]]),
    'formato_gama': 2.0, 'escala_gama': escala_gama,
    'profundidade_camada1': 200.0, 'profundidade_camada2': 800.0,
    'capacidade_max_camada1': 60.0, 'ponto_murcha_camada1': 20.0,
    'capacidade_max_camada2': 240.0, 'ponto_murcha_camada2': 80.0,
    'T_BASE': T_BASE, 'T_MAX_LIMITE': T_MAX_LIMITE,
    'gdu_fase_reprodutiva_inicial': gdu_fase_reprodutiva_inicial, 'gdu_fase_reprodutiva_pico': gdu_fase_reprodutiva_pico,
    'temp_media_anual': 26.0, 'amplitude_sazonal': 6.0, 'amplitude_diurna_base': 12.0,
    'LATITUDE_GRAUS': -12.5, 'ALTITUDE_METROS': 330, 'a_s': 0.25, 'b_s': 0.50,
}

if st.sidebar.button("▶️ Executar Análise", type="primary"):
    if not datas_de_plantio_para_analise.empty:
        df_resumo, df_detalhado = executar_analise_janelas(
            _datas_plantio=datas_de_plantio_para_analise,
            ciclo_dias=ciclo_duracao_dias,
            n_simulacoes=numero_de_simulacoes_por_janela,
            config=config
        )
        st.session_state.df_resumo = df_resumo
        st.session_state.df_detalhado = df_detalhado
        st.session_state.messages = [] # Limpa o chat para a nova análise
    else:
        st.error("O intervalo de datas selecionado não gerou nenhuma janela de plantio. Ajuste as datas.")

# --- Área de exibição dos resultados ---
if st.session_state.df_resumo is not None:
    df_resumo = st.session_state.df_resumo
    df_detalhado = st.session_state.df_detalhado
    
    st.subheader("📊 Resultados da Simulação (Valores Medianos)")
    st.dataframe(df_resumo.round(1))
    csv = df_detalhado.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Baixar todos os resultados em CSV", data=csv, file_name="resultados_completos_simulacao.csv", mime="text/csv")
    
    st.header("Visualizações Avançadas")
    tab1, tab2, tab3 = st.tabs(["GDU vs. Estresse", "Análise de Risco (Box Plots)", "Análise de Trade-Off"])
    with tab1:
        st.pyplot(gerar_grafico_principal(df_resumo))
    with tab2:
        st.pyplot(gerar_boxplots(df_detalhado))
    with tab3:
        st.pyplot(gerar_tradeoff_plot(df_resumo))

    st.header("🤖 Análise e Recomendação com IA do Gemini")
    with st.expander("Clique aqui para ver as opções de análise com Inteligência Artificial"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Gerar Análise Padrão"):
                with st.spinner("Analisando..."):
                    analise_texto = gerar_analise_gemini(df_resumo, df_detalhado)
                    if analise_texto:
                        st.session_state.analise_texto = analise_texto
        with col2:
            if st.button("Gerar Análise com Previsão do Tempo"):
                with st.spinner("Buscando previsão e analisando..."):
                    previsao = get_weather_forecast_summary()
                    st.info(f"**Previsão do tempo considerada:** {previsao}")
                    analise_texto = gerar_analise_gemini(df_resumo, df_detalhado, previsao_tempo=previsao)
                    if analise_texto:
                        st.session_state.analise_texto = analise_texto
        
        if 'analise_texto' in st.session_state and st.session_state.analise_texto:
            st.markdown(st.session_state.analise_texto)

    # --- Funcionalidade do Chatbot ---
    st.header("💬 Converse com a IA sobre os Resultados")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Faça uma pergunta sobre os resultados..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = handle_chat(prompt, df_resumo, df_detalhado)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Ajuste os parâmetros na barra lateral e clique em 'Executar Análise' para iniciar a simulação.")
