import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gamma
from datetime import timedelta, datetime
from typing import Dict, Any, Tuple, List
from numpy.typing import NDArray
import json

# =============================================================================
# SEÇÃO 0: CONSTANTES E CONFIGURAÇÃO GLOBAL
# =============================================================================
MODEL_CONSTANTS = {
    'STEFAN_BOLTZMANN': 4.903e-9, 'PRESSAO_ATM_PADRAO': 101.3, 'COEF_PSICROMETRICO': 0.000665,
    'FLUXO_CALOR_SOLO': 0, 'CONSTANTE_TEMP_K_C': 273.16, 'CONSTANTE_GAS_VAPOR_AGUA': 0.0820,
    'COEF_TEMP_TERMO_2_PM': 900, 'COEF_VENDO_TERMO_2_PM': 0.34, 'ALBEDO_PADRAO': 0.23,
    'COEF_ANGSTROM_A': 0.25, 'COEF_ANGSTROM_B': 0.50,
}
MONTHLY_CLIMATE_PARAMS = {
    1: {'p_chuva': [0.20, 0.75], 'formato_gama': 2.5, 'escala_gama': 12.0}, 2: {'p_chuva': [0.22, 0.78], 'formato_gama': 2.5, 'escala_gama': 11.5},
    3: {'p_chuva': [0.25, 0.70], 'formato_gama': 2.2, 'escala_gama': 11.0}, 4: {'p_chuva': [0.45, 0.55], 'formato_gama': 2.0, 'escala_gama': 9.0},
    5: {'p_chuva': [0.75, 0.30], 'formato_gama': 1.8, 'escala_gama': 6.0}, 6: {'p_chuva': [0.90, 0.20], 'formato_gama': 1.5, 'escala_gama': 5.0},
    7: {'p_chuva': [0.92, 0.18], 'formato_gama': 1.5, 'escala_gama': 4.5}, 8: {'p_chuva': [0.88, 0.25], 'formato_gama': 1.8, 'escala_gama': 5.5},
    9: {'p_chuva': [0.70, 0.40], 'formato_gama': 2.0, 'escala_gama': 7.0}, 10: {'p_chuva': [0.40, 0.60], 'formato_gama': 2.2, 'escala_gama': 9.5},
    11: {'p_chuva': [0.30, 0.70], 'formato_gama': 2.4, 'escala_gama': 10.5}, 12: {'p_chuva': [0.20, 0.80], 'formato_gama': 2.5, 'escala_gama': 12.5},
}

# =============================================================================
# SEÇÃO 1: FUNÇÕES DO MODELO DE SIMULAÇÃO (Lógica Principal)
# =============================================================================
def calcular_kc_diario(gdu_acumulado: float, cfg: Dict[str, Any]) -> float:
    """Calcula o Coeficiente da Cultura (Kc) diário com base nos estágios fenológicos (GDU)."""
    if gdu_acumulado <= cfg['gdu_fim_fase_inicial']: return cfg['kc_ini']
    elif gdu_acumulado <= cfg['gdu_fim_fase_desenvolvimento']:
        gdu_fase_dev = gdu_acumulado - cfg['gdu_fim_fase_inicial']
        gdu_total_dev = cfg['gdu_fim_fase_desenvolvimento'] - cfg['gdu_fim_fase_inicial']
        return cfg['kc_ini'] + (cfg['kc_mid'] - cfg['kc_ini']) * (gdu_fase_dev / gdu_total_dev)
    elif gdu_acumulado <= cfg['gdu_fim_fase_media']: return cfg['kc_mid']
    elif gdu_acumulado < cfg['gdu_maturacao']:
        gdu_fase_final = gdu_acumulado - cfg['gdu_fim_fase_media']
        gdu_total_final = cfg['gdu_maturacao'] - cfg['gdu_fim_fase_media']
        return cfg['kc_mid'] + (cfg['kc_end'] - cfg['kc_mid']) * (gdu_fase_final / gdu_total_final)
    else: return cfg['kc_end']

def atualizar_parametros_cultura(gdu_acumulado: float, cfg: Dict[str, Any]) -> Tuple[float, float]:
    """Atualiza dinamicamente o albedo e a profundidade da raiz com base no GDU."""
    if gdu_acumulado < cfg['gdu_fim_fase_desenvolvimento']:
        albedo = 0.20; fator_crescimento_raiz = gdu_acumulado / cfg['gdu_fim_fase_desenvolvimento']
        prof_raiz_efetiva = cfg['profundidade_camada1'] + (cfg['profundidade_camada2'] * fator_crescimento_raiz)
    else:
        albedo = 0.23; prof_raiz_efetiva = cfg['profundidade_camada1'] + cfg['profundidade_camada2']
    prof_max = cfg['profundidade_camada1'] + cfg['profundidade_camada2']; return albedo, min(prof_raiz_efetiva, prof_max)

def simular_processos_diarios(n_dias: int, start_day_of_year: int, latitude_rad: float, altitude: float, cfg: Dict[str, Any]) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Executa a simulação diária para uma safra completa."""
    precipitacao, eto_arr, gdu_padrao, gdu_ajustado, dias_estresse, runoff = (np.zeros(n_dias) for _ in range(6))
    estado_chuva = np.zeros(n_dias, dtype=np.int64); armazenamento_c1, armazenamento_c2 = (np.zeros(n_dias) for _ in range(2))
    armazenamento_c1[0] = cfg['capacidade_max_camada1'] * 0.8; armazenamento_c2[0] = cfg['capacidade_max_camada2'] * 0.8
    gdu_acumulado = 0.0; current_date = datetime(2025, 1, 1) + timedelta(days=start_day_of_year - 1)
    for i in range(1, n_dias):
        month = current_date.month; params_mes = cfg['CLIMA_MENSAL'][str(month)]
        p_seco_chuvoso, p_chuvoso_chuvoso = params_mes['p_chuva']; matriz_transicao_chuva = np.array([[1 - p_seco_chuvoso, p_seco_chuvoso], [1 - p_chuvoso_chuvoso, p_chuvoso_chuvoso]])
        if np.random.rand() < matriz_transicao_chuva[estado_chuva[i-1], 1]: estado_chuva[i] = 1
        precipitacao[i] = gamma.rvs(a=params_mes['formato_gama'], scale=params_mes['escala_gama']) if estado_chuva[i] == 1 else 0.0
        day_index = start_day_of_year + i; current_date = datetime(2025, 1, 1) + timedelta(days=day_index - 1)
        temp_media_sazonal = cfg['temp_media_anual'] + cfg['amplitude_sazonal'] * np.sin(2 * np.pi * (day_index - 80) / 365.25)
        amp_diurna = cfg['amplitude_diurna_base'] * (0.7 if estado_chuva[i] == 1 else 1.0)
        t_max = temp_media_sazonal + amp_diurna / 2 + np.random.normal(0, 1.5); t_min = t_max - amp_diurna - np.random.normal(0, 1.5)
        t_mean = (t_max + t_min) / 2; albedo, prof_raiz_efetiva = atualizar_parametros_cultura(gdu_acumulado, cfg)
        P = cfg['PRESSAO_ATM_PADRAO'] * ((293 - 0.0065 * altitude) / 293)**5.26; gamma_p = cfg['COEF_PSICROMETRICO'] * P
        e_tmax = 0.6108 * np.exp((17.27 * t_max) / (t_max + 237.3)); e_tmin = 0.6108 * np.exp((17.27 * t_min) / (t_min + 237.3))
        es = (e_tmax + e_tmin) / 2; ea = (np.random.uniform(75, 95) / 100) * es if estado_chuva[i] == 1 else (np.random.uniform(45, 70) / 100) * es
        delta_v = 4098 * (0.6108 * np.exp((17.27 * t_mean) / (t_mean + 237.3))) / (t_mean + 237.3)**2
        dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * day_index); delta_sol = 0.409 * np.sin(2 * np.pi / 365 * day_index - 1.39)
        omega_s = np.arccos(-np.tan(latitude_rad) * np.tan(delta_sol)); Ra = (24 * 60 / np.pi) * cfg['CONSTANTE_GAS_VAPOR_AGUA'] * dr * (omega_s * np.sin(latitude_rad) * np.sin(delta_sol) + np.cos(latitude_rad) * np.cos(delta_sol) * np.sin(omega_s))
        Rs = Ra * (cfg['COEF_ANGSTROM_A'] + cfg['COEF_ANGSTROM_B'] * (0.40 if estado_chuva[i] == 1 else 0.8)); Rso = (0.75 + 2e-5 * altitude) * Ra
        Rns = (1 - albedo) * Rs; t_max_k4 = (t_max + cfg['CONSTANTE_TEMP_K_C'])**4; t_min_k4 = (t_min + cfg['CONSTANTE_TEMP_K_C'])**4
        Rnl = cfg['STEFAN_BOLTZMANN'] * ((t_max_k4 + t_min_k4) / 2) * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * (Rs / Rso) - 0.35); Rn = Rns - Rnl
        u2 = np.random.uniform(1.5, 3.5); termo1 = (0.408 * delta_v * (Rn - cfg['FLUXO_CALOR_SOLO'])); termo2 = (gamma_p * (cfg['COEF_TEMP_TERMO_2_PM'] / (t_mean + cfg['CONSTANTE_TEMP_K_C'])) * u2 * (es - ea))
        denominador = (delta_v + gamma_p * (1 + cfg['COEF_VENDO_TERMO_2_PM'] * u2)); eto_arr[i] = max(0, (termo1 + termo2) / denominador)
        kc_diario = calcular_kc_diario(gdu_acumulado, cfg); etc_diaria = kc_diario * eto_arr[i]
        agua_c1_inicio = armazenamento_c1[i-1] + precipitacao[i]; percolacao = max(0, agua_c1_inicio - cfg['capacidade_max_camada1'])
        armazenamento_c1_temp = agua_c1_inicio - percolacao; agua_c2_inicio = armazenamento_c2[i-1] + percolacao
        runoff[i] = max(0, agua_c2_inicio - cfg['capacidade_max_camada2']); armazenamento_c2_temp = agua_c2_inicio - runoff[i]
        agua_disponivel_c1 = max(0, armazenamento_c1_temp - cfg['ponto_murcha_camada1']); agua_disponivel_c2 = max(0, armazenamento_c2_temp - cfg['ponto_murcha_camada2'])
        fator_extracao_c1 = cfg['profundidade_camada1'] / prof_raiz_efetiva if prof_raiz_efetiva > 0 else 1
        demanda_c1 = etc_diaria * fator_extracao_c1; demanda_c2 = etc_diaria * (1 - fator_extracao_c1)
        etr_c1 = min(agua_disponivel_c1, demanda_c1); etr_c2 = min(agua_disponivel_c2, demanda_c2) if prof_raiz_efetiva > cfg['profundidade_camada1'] else 0
        armazenamento_c1[i] = armazenamento_c1_temp - etr_c1; armazenamento_c2[i] = armazenamento_c2_temp - etr_c2
        gdu_padrao[i] = max(0, (min(t_max, cfg['T_MAX_LIMITE']) + max(t_min, cfg['T_BASE']))/2 - cfg['T_BASE'])
        cap_disponivel_total_raiz = (cfg['capacidade_max_camada1'] - cfg['ponto_murcha_camada1']) + (cfg['capacidade_max_camada2'] - cfg['ponto_murcha_camada2']) * min(1, max(0, (prof_raiz_efetiva - cfg['profundidade_camada1']) / cfg['profundidade_camada2']))
        agua_disponivel_total_raiz = max(0, armazenamento_c1[i] - cfg['ponto_murcha_camada1']) + max(0, armazenamento_c2[i] - cfg['ponto_murcha_camada2'])
        fator_umidade_relativa = agua_disponivel_total_raiz / cap_disponivel_total_raiz if cap_disponivel_total_raiz > 0 else 0
        umbral_estresse = 0.45; fator_estresse = min(1.0, fator_umidade_relativa / umbral_estresse) if fator_umidade_relativa < umbral_estresse else 1.0
        if fator_estresse < 1.0: dias_estresse[i] = 1
        gdu_ajustado[i] = gdu_padrao[i] * fator_estresse; gdu_acumulado += gdu_ajustado[i]
    return gdu_ajustado, dias_estresse, precipitacao, runoff

# =============================================================================
# SEÇÃO 2: FUNÇÕES PARA EXECUÇÃO DA ANÁLISE E VISUALIZAÇÃO
# =============================================================================
@st.cache_data
def executar_analise_janelas(
    datas_plantio_str: List[str],
    ciclo_dias: int,
    n_simulacoes: int,
    config_json: str
) -> pd.DataFrame:
    """Executa a análise de Monte Carlo e retorna o DataFrame completo com todos os resultados."""
    config = json.loads(config_json)
    datas_plantio = pd.to_datetime(datas_plantio_str)
    lista_resultados_completos = []
    barra_progresso = st.progress(0, text="Iniciando simulação...")
    status_texto = st.empty()
    for i, data_inicio in enumerate(datas_plantio):
        data_fim = data_inicio + timedelta(days=ciclo_dias - 1)
        status_texto.text(f"Simulando para a janela de plantio: {data_inicio.strftime('%d-%m-%Y')} a {data_fim.strftime('%d-%m-%Y')}")
        resultados_monte_carlo = []
        for sim in range(n_simulacoes):
            start_day_of_year = data_inicio.dayofyear; latitude_rad = np.deg2rad(config['latitude'])
            gdu_adj, estresse_dia, precip_dia, runoff_dia = simular_processos_diarios(
                ciclo_dias, start_day_of_year, latitude_rad, config['altitude'], config
            )
            metricas = {
                'gdu_final_ajustado': np.sum(gdu_adj), 'dias_estresse_hidrico': np.sum(estresse_dia),
                'precipitacao_total_safra': np.sum(precip_dia), 'runoff_total_safra': np.sum(runoff_dia)
            }
            resultados_monte_carlo.append(metricas)
        df_mc = pd.DataFrame(resultados_monte_carlo)
        df_mc['data_plantio'] = data_inicio.strftime('%Y-%m-%d')
        lista_resultados_completos.append(df_mc)
        barra_progresso.progress((i + 1) / len(datas_plantio), text=f"Analisando janela {i+1}/{len(datas_plantio)}")
    status_texto.text("Simulação concluída!")
    return pd.concat(lista_resultados_completos, ignore_index=True)

def gerar_graficos_distribuicao_plotly(df_completo: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """Gera box plots interativos para GDU e Dias de Estresse Hídrico."""
    df_completo_sorted = df_completo.sort_values(by='data_plantio')
    fig_gdu = px.box(df_completo_sorted, x='data_plantio', y='gdu_final_ajustado',
                     title="Distribuição do GDU Final Ajustado por Janela de Plantio",
                     labels={"data_plantio": "Data de Plantio", "gdu_final_ajustado": "GDU Final Acumulado"},
                     color_discrete_sequence=['cornflowerblue'])
    fig_gdu.update_layout(title_x=0.5, xaxis_title=None, xaxis={'tickangle': 45})
    fig_estresse = px.box(df_completo_sorted, x='data_plantio', y='dias_estresse_hidrico',
                          title="Distribuição dos Dias de Estresse Hídrico por Janela de Plantio",
                          labels={"data_plantio": "Data de Plantio", "dias_estresse_hidrico": "Total de Dias com Estresse"},
                          color_discrete_sequence=['tomato'])
    fig_estresse.update_layout(title_x=0.5, xaxis_title="Data de Plantio", xaxis={'tickangle': 45})
    return fig_gdu, fig_estresse

def gerar_grafico_tradeoff_plotly(df_medianas: pd.DataFrame) -> go.Figure:
    """Gera um gráfico de dispersão para visualizar o trade-off entre GDU e estresse."""
    df_medianas.index = pd.to_datetime(df_medianas.index) # <-- CORREÇÃO APLICADA AQUI
    df_medianas['data_plantio_str'] = df_medianas.index.strftime('%d/%m')
    fig = px.scatter(df_medianas, x='dias_estresse_hidrico', y='gdu_final_ajustado',
                     text='data_plantio_str', title="Análise de Trade-Off: GDU vs. Estresse Hídrico",
                     labels={'dias_estresse_hidrico': 'Dias de Estresse Hídrico (Mediana)', 'gdu_final_ajustado': 'GDU Final Ajustado (Mediana)'},
                     hover_data={'data_plantio_str': False, 'dias_estresse_hidrico': ':.1f', 'gdu_final_ajustado': ':.1f'})
    fig.update_traces(textposition='top center', marker=dict(size=12, color='mediumpurple'), textfont_size=11)
    fig.update_layout(title_x=0.5)
    return fig

# =============================================================================
# SEÇÃO 3: INTERFACE DO USUÁRIO (Streamlit App)
# =============================================================================
st.set_page_config(layout="wide", page_title="Otimização da Janela de Plantio")
st.title("🌽 Dashboard de Otimização da Janela de Plantio")
st.markdown("""
Esta ferramenta utiliza uma simulação de Monte Carlo para analisar diferentes janelas de plantio.
O objetivo é identificar as datas que maximizam o acúmulo de Graus-Dia (GDU), minimizando os dias de estresse hídrico.
Os **gráficos de distribuição (box plot)** abaixo mostram não apenas a média (mediana), mas também a variabilidade e o risco (outliers) de cada cenário.
""")

st.sidebar.header("Parâmetros da Análise")
st.sidebar.subheader("🗓️ Janelas de Plantio")
data_inicial_analise = st.sidebar.date_input("Data Inicial da Análise", datetime(2025, 9, 15))
data_final_analise = st.sidebar.date_input("Data Final da Análise", datetime(2025, 12, 15))
intervalo_dias = st.sidebar.slider("Intervalo entre datas (dias)", 5, 20, 10)
datas_de_plantio_para_analise = pd.to_datetime(pd.date_range(start=data_inicial_analise, end=data_final_analise, freq=f'{intervalo_dias}D'))
st.sidebar.subheader("⚙️ Parâmetros da Simulação")
ciclo_duracao_dias = st.sidebar.slider("Duração do Ciclo da Cultura (dias)", 90, 180, 140)
numero_de_simulacoes_por_janela = st.sidebar.slider("Nº de Simulações por Janela", 50, 500, 150)
with st.sidebar.expander("🔬 Parâmetros Avançados do Modelo Agroclimático", expanded=True):
    st.markdown("**Localização e Clima Base**")
    latitude = st.number_input("Latitude (graus decimais)", -90.0, 90.0, -12.8, 0.1)
    altitude = st.number_input("Altitude (metros)", 0, 4000, 430, 10)
    st.markdown("**Parâmetros da Cultura (GDU)**")
    T_BASE = st.number_input("Temperatura Base (°C)", 5.0, 15.0, 10.0, 0.5)
    T_MAX_LIMITE = st.number_input("Temperatura Máxima Limite (°C)", 28.0, 45.0, 34.0, 0.5)
    st.markdown("**Parâmetros da Curva Kc (Coeficiente da Cultura)**")
    kc_ini, kc_mid, kc_end = st.number_input("Kc Inicial", 0.1, 0.8, 0.4, 0.05), st.number_input("Kc Médio (pico)", 0.8, 1.5, 1.2, 0.05), st.number_input("Kc Final", 0.2, 1.0, 0.5, 0.05)
    st.markdown("**Estágios Fenológicos (baseado em GDU acumulado)**")
    gdu_fim_fase_inicial = st.number_input("GDU para Fim da Fase Inicial", 100.0, 300.0, 180.0, 10.0)
    gdu_fim_fase_desenvolvimento = st.number_input("GDU para Fim da Fase de Desenvolvimento", 400.0, 900.0, 750.0, 10.0)
    gdu_fim_fase_media = st.number_input("GDU para Fim da Fase Média", 900.0, 1500.0, 1250.0, 10.0)
    gdu_maturacao = st.number_input("GDU para Maturação Fisiológica", 1300.0, 2000.0, 1600.0, 10.0)

config = {
    'latitude': latitude, 'altitude': altitude, 'T_BASE': T_BASE, 'T_MAX_LIMITE': T_MAX_LIMITE,
    'kc_ini': kc_ini, 'kc_mid': kc_mid, 'kc_end': kc_end,
    'gdu_fim_fase_inicial': gdu_fim_fase_inicial, 'gdu_fim_fase_desenvolvimento': gdu_fim_fase_desenvolvimento,
    'gdu_fim_fase_media': gdu_fim_fase_media, 'gdu_maturacao': gdu_maturacao,
    'profundidade_camada1': 200.0, 'profundidade_camada2': 800.0, 'capacidade_max_camada1': 60.0,
    'ponto_murcha_camada1': 20.0, 'capacidade_max_camada2': 240.0, 'ponto_murcha_camada2': 80.0,
    'temp_media_anual': 25.0, 'amplitude_sazonal': 5.0, 'amplitude_diurna_base': 12.0,
    'CLIMA_MENSAL': MONTHLY_CLIMATE_PARAMS, **MODEL_CONSTANTS
}

if st.button("▶️ Executar Análise de Janelas de Plantio"):
    if not datas_de_plantio_para_analise.empty:
        with st.spinner('Executando simulações... Isso pode levar alguns minutos.'):
            config_string_hasheavel = json.dumps(config, sort_keys=True)
            datas_plantio_string_list = [d.strftime('%Y-%m-%d') for d in datas_de_plantio_para_analise]
            df_resultados_completos = executar_analise_janelas(
                datas_plantio_str=datas_plantio_string_list,
                ciclo_dias=ciclo_duracao_dias,
                n_simulacoes=numero_de_simulacoes_por_janela,
                config_json=config_string_hasheavel
            )
        st.success("Análise concluída com sucesso!")
        df_medianas = df_resultados_completos.groupby('data_plantio').median()
        st.subheader("📊 Resultados da Simulação (Valores Medianos)")
        st.dataframe(df_medianas.round(1))
        csv_data = df_resultados_completos.to_csv(index=False).encode('utf-8')
        st.download_button(
           label="📥 Baixar todos os resultados em CSV",
           data=csv_data, file_name='resultados_completos_simulacao.csv', mime='text/csv',
        )
        st.subheader("📈 Análise de Risco e Variabilidade")
        st.markdown("Os gráficos de caixa (box plot) mostram a mediana (linha central), os quartis (caixa) e a faixa de resultados (bigodes) para cada janela. Pontos individuais são outliers e indicam cenários extremos.")
        fig_gdu, fig_estresse = gerar_graficos_distribuicao_plotly(df_resultados_completos)
        st.plotly_chart(fig_gdu, use_container_width=True)
        st.plotly_chart(fig_estresse, use_container_width=True)
        st.subheader("⚖️ Análise de Trade-Off")
        st.markdown("Este gráfico de dispersão ajuda a visualizar a melhor relação GDU vs. Estresse. Janelas ideais se localizam no canto **superior esquerdo** (alto GDU, baixo estresse).")
        fig_tradeoff = gerar_grafico_tradeoff_plotly(df_medianas)
        st.plotly_chart(fig_tradeoff, use_container_width=True)
        st.subheader("💡 Análise e Conclusão")
        melhor_gdu_data = df_medianas['gdu_final_ajustado'].idxmax()
        melhor_gdu_valor = df_medianas['gdu_final_ajustado'].max()
        menor_estresse_data = df_medianas['dias_estresse_hidrico'].idxmin()
        menor_estresse_valor = df_medianas['dias_estresse_hidrico'].min()
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Melhor Janela para GDU Máximo (Mediana)", value=datetime.strptime(melhor_gdu_data, '%Y-%m-%d').strftime('%d/%m/%Y'), delta=f"{melhor_gdu_valor:.1f} GDU")
            st.markdown(f"A data de plantio em **{datetime.strptime(melhor_gdu_data, '%Y-%m-%d').strftime('%d/%m/%Y')}** resultou no maior acúmulo mediano de Graus-Dia.")
        with col2:
            st.metric(label="Melhor Janela para Estresse Mínimo (Mediana)", value=datetime.strptime(menor_estresse_data, '%Y-%m-%d').strftime('%d/%m/%Y'), delta=f"{menor_estresse_valor:.0f} dias de estresse", delta_color="inverse")
            st.markdown(f"A data de plantio em **{datetime.strptime(menor_estresse_data, '%Y-%m-%d').strftime('%d/%m/%Y')}** apresentou a menor quantidade mediana de dias sob estresse hídrico.")
        if melhor_gdu_data == menor_estresse_data:
            st.info(f"🏆 **Recomendação:** A data de **{datetime.strptime(melhor_gdu_data, '%Y-%m-%d').strftime('%d/%m/%Y')}** parece ser a ideal, pois maximiza o GDU e minimiza o estresse hídrico simultaneamente, com base nas medianas.")
        else:
            st.warning("⚠️ **Atenção:** Existe um trade-off. Use os gráficos acima para avaliar o risco de cada data e tomar a decisão final ponderando os riscos hídricos versus o potencial produtivo.")
    else:
        st.error("O intervalo de datas selecionado não gerou nenhuma janela de plantio. Por favor, ajuste as datas inicial e final.")
