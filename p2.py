import os
import io
import zipfile
import textwrap
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import sklearn
from packaging.version import Version
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# suppress sklearn InconsistentVersionWarning (aparece quando pipelines foram salvos em outra versão)
# se preferir ver os avisos, comente a linha abaixo.
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

# -------------------------
# OneHotEncoder compatível com versões sklearn
# -------------------------
def make_onehot(handle_unknown='ignore'):
    """
    Retorna um OneHotEncoder compatível com a versão instalada do scikit-learn.
    Versões >= 1.2 usam 'sparse_output'; versões < 1.2 usam 'sparse'.
    """
    try:
        # usar packaging.version.Version para comparar versões
        if Version(sklearn.__version__) >= Version("1.2"):
            return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
        else:
            return OneHotEncoder(handle_unknown=handle_unknown, sparse=False)
    except Exception:
        # fallback genérico caso algo inesperado aconteça
        try:
            return OneHotEncoder(handle_unknown=handle_unknown)
        except Exception:
            return OneHotEncoder()

# -------------------------
# Config / constantes
# -------------------------
st.set_page_config(page_title="ENEM 2024 — Explorador, Modelador e Análise Regional", layout="wide")
DEFAULT_ENEM = os.path.join("data", "raw", "Enem_2024_Amostra_Perfeita.xlsx")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Colunas de notas esperadas (normalizadas)
ENEM_NOTAS_LOWER = [
    'nota_cn_ciencias_da_natureza',
    'nota_ch_ciencias_humanas',
    'nota_lc_linguagens_e_codigos',
    'nota_mt_matematica',
    'nota_redacao',
    'nota_media_5_notas'
]

# -------------------------
# Helpers: carga e session_state
# -------------------------
@st.cache_data(show_spinner="Carregando ENEM do disco...")
def load_enem_from_disk(path: str = DEFAULT_ENEM) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_excel(path, engine='openpyxl')
    df.columns = [str(c).lower().strip() for c in df.columns]
    return df

def read_uploaded_excel(uploaded) -> pd.DataFrame:
    try:
        df = pd.read_excel(uploaded, engine='openpyxl')
        df.columns = [str(c).lower().strip() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Erro ao ler arquivo enviado: {e}")
        return pd.DataFrame()

def ensure_session_df():
    """Garante que st.session_state['df_enem'] exista (carrega do disco se disponível)."""
    if 'df_enem' not in st.session_state:
        df_disk = load_enem_from_disk(DEFAULT_ENEM)
        if not df_disk.empty:
            st.session_state['df_enem'] = df_disk
        else:
            st.session_state['df_enem'] = pd.DataFrame()

def summary_stats(df, cols):
    desc = df[cols].describe().T
    desc['missing'] = df[cols].isna().sum().values
    desc = desc[['count', 'missing', 'mean', '50%', 'std', 'min', 'max']]
    desc = desc.rename(columns={'50%': 'median'})
    return desc

# -------------------------
# Modelagem: pré-processamento, treino, serialização
# -------------------------
def build_numeric_transformer():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

def train_and_evaluate_models(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42, model_dir=MODEL_DIR):
    """
    Treina modelos (RandomForest, LinearRegression, SVR) e salva pipelines.
    Retorna dicionário com métricas, paths e melhor modelo (por R2).
    """
    os.makedirs(model_dir, exist_ok=True)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', build_numeric_transformer(), num_cols),
        ('cat', make_onehot(handle_unknown='ignore'), cat_cols)
    ], remainder='drop')

    models = {
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=random_state),
        'LinearRegression': LinearRegression(),
        'SVR': SVR()
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    metrics = {}
    model_paths = {}

    for name, estimator in models.items():
        pipe = Pipeline([('preproc', preprocessor), ('model', estimator)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        metrics[name] = {'r2': float(r2), 'mse': float(mse)}
        path = os.path.join(model_dir, f'pipeline_{name}.joblib')
        joblib.dump(pipe, path)
        model_paths[name] = path

    # escolher melhor por R2 (maior). Se empate em R2, escolhe menor MSE.
    sorted_models = sorted(metrics.items(), key=lambda x: (-x[1]['r2'], x[1]['mse']))
    best_name = sorted_models[0][0]
    best_path = model_paths[best_name]
    best_pipeline = joblib.load(best_path)
    joblib.dump(best_pipeline, os.path.join(model_dir, 'best_pipeline.joblib'))

    # salvar metadados para predição interativa
    feature_names = X.columns.tolist()
    feature_defaults = X.median().to_dict()
    joblib.dump(feature_names, os.path.join(model_dir, 'feature_names.joblib'))
    joblib.dump(feature_defaults, os.path.join(model_dir, 'feature_defaults.joblib'))

    return {'metrics': metrics, 'best_model': best_name, 'model_paths': model_paths, 'best_path': best_path}

def load_best_pipeline(model_dir=MODEL_DIR):
    best_pipeline_path = os.path.join(model_dir, 'best_pipeline.joblib')
    if os.path.exists(best_pipeline_path):
        return joblib.load(best_pipeline_path)
    return None

def load_feature_metadata(model_dir=MODEL_DIR):
    fn_path = os.path.join(model_dir, 'feature_names.joblib')
    fd_path = os.path.join(model_dir, 'feature_defaults.joblib')
    if os.path.exists(fn_path) and os.path.exists(fd_path):
        return joblib.load(fn_path), joblib.load(fd_path)
    return None, None

# -------------------------
# UI: uploader top-level + abas
# -------------------------
st.markdown("<h1 style='color:#0b4a6f'>ENEM 2024 — Explorador, Modelador e Análise Regional</h1>", unsafe_allow_html=True)
st.markdown("#### Ferramenta integrada: explore as notas do ENEM 2024, treine modelos e teste relações entre notas e desenvolvimento regional.")
st.write("Leia a introdução detalhada na primeira aba. Coloque `data/raw/Enem_2024_Amostra_Perfeita.xlsx` ou faça upload no topo (visível em todas as abas).")

ensure_session_df()

col_up1, col_up2 = st.columns([3,1])
with col_up1:
    uploaded = st.file_uploader("Upload: Enem_2024_Amostra_Perfeita.xlsx (opcional) — sobrescreve sessão", type=["xlsx"])
with col_up2:
    if st.button("Limpar dataset carregado"):
        st.session_state['df_enem'] = pd.DataFrame()
        st.success("Dataset em sessão limpo.")

if uploaded is not None:
    df_uploaded = read_uploaded_excel(uploaded)
    if not df_uploaded.empty:
        st.session_state['df_enem'] = df_uploaded
        st.success("Arquivo carregado e salvo na sessão (disponível em todas as abas).")

# create five tabs now (keeps original four + new downloads tab)
tab_intro, tab_eda, tab_model, tab_regions, tab_downloads = st.tabs([
    "Introdução (guia)",
    "EDA Interativo (Notas)",
    "Modelagem & Previsão",
    "Pesquisa Regiões (Análise)",
    "Downloads / MLOps"
])

# -------------------------
# TAB 1 — Introdução (detalhada)
# -------------------------
with tab_intro:
    st.header("Guia detalhado do aplicativo — contexto e objetivo")
    st.markdown(textwrap.dedent("""
    **Contexto e tema**
    - O ENEM (Exame Nacional do Ensino Médio) produz notas que são indicadores de desempenho educacional por participante / município.
    - Indicadores de desenvolvimento econômico e social (PIB per capita, IDH/IDHM) tendem a se relacionar com o desempenho educacional: regiões mais desenvolvidas, em média, têm melhores condições de ensino.

    **Objetivo do app**
    - Investigar, de forma exploratória e reprodutível, se as **notas do ENEM** na amostra disponibilizada estão associadas a regiões mais ou menos desenvolvidas.
    - Permitir treinar modelos preditivos para uma nota-alvo (por exemplo, `nota_media_5_notas`) e comparar desempenho entre modelos.
    - Oferecer uma análise prática: verificar se as 10 maiores notas vêm majoritariamente de regiões "mais desenvolvidas", se as 10 menores vêm de "menos desenvolvidas", etc.

    **Hipótese testada (exploratória)**
    - Hipótese nula (H0): não há diferença sistemática entre notas médias por nível de desenvolvimento regional.
    - Hipótese alternativa (H1): regiões mais desenvolvidas têm, em média, notas de ENEM maiores.

    **Observações importantes**
    - Esta ferramenta é **exploratória** — conclusões definitivas demandam amostragem representativa e testes estatísticos formais.
    - A classificação "mais/menos desenvolvidos" aqui usa uma **heurística macro-regional** (SE/S/CO → mais; N/NE → menos). Posso integrar PIB per capita ou IDHM para uma classificação mais precisa se desejar.
    - Quando você usar a predição interativa, o app tentará **estimar heurísticamente** em qual UF (estado) a nota prevista faria mais sentido (com base nas médias por UF na sua amostra). Isso é uma suposição simples, não uma inferência causal.
    """))

# -------------------------
# TAB 2 — EDA Interativo
# -------------------------
with tab_eda:
    st.header("EDA Interativo — como usar")
    st.markdown("Nesta aba você explora a base com filtros (ano, UF, município). Os gráficos são interativos — passe o cursor para ver detalhes.")
    df = st.session_state.get('df_enem', pd.DataFrame())
    if df.empty:
        df_disk = load_enem_from_disk(DEFAULT_ENEM)
        if not df_disk.empty:
            st.session_state['df_enem'] = df_disk
            df = df_disk

    if df.empty:
        st.info("Sem dataset disponível. Faça upload no topo ou coloque `data/raw/Enem_2024_Amostra_Perfeita.xlsx` no projeto.")
        st.stop()

    st.subheader("Visão dos dados carregados")
    st.write("Abaixo mostramos as primeiras colunas detectadas (já normalizadas em lowercase):")
    st.write(list(df.columns)[:200])

    # DETECÇÃO ROBUSTA DE COLUNAS DE NOTA:
    notas_presentes = [c for c in ENEM_NOTAS_LOWER if c in df.columns]
    # fallback 1: procurar qualquer coluna contendo 'nota' no nome
    if not notas_presentes:
        notas_presentes = [c for c in df.columns if 'nota' in c.lower()]
    # fallback 2: usar colunas numéricas plausíveis (5 notas + redação) se ainda vazio
    if not notas_presentes:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # heurística simples: pegar até 6 primeiras numéricas que contenham 'mt','lc','red','cn','ch' ou 'media'
        preferred_tokens = ['mt', 'lc', 'red', 'redacao', 'cn', 'ch', 'media', 'nota']
        picked = [c for c in numeric_cols if any(tok in c.lower() for tok in preferred_tokens)]
        notas_presentes = picked[:6] if picked else numeric_cols[:6]

    st.write(f"Notas detectadas automaticamente (heurística): {notas_presentes}")

    for c in notas_presentes + ['longitude','latitude','nu_ano']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    st.session_state['df_enem'] = df  # salva versões normalizadas

    st.sidebar.header("Filtros & Controles (EDA)")
    years = sorted(df['nu_ano'].dropna().unique().astype(int).tolist()) if 'nu_ano' in df.columns else []
    selected_year = st.sidebar.selectbox("Filtrar por ano (nu_ano)", options=[None] + years, index=0)
    uf_opts = sorted(df['sg_uf_prova'].dropna().unique().tolist()) if 'sg_uf_prova' in df.columns else []
    selected_ufs = st.sidebar.multiselect("Filtrar por UF (sg_uf_prova)", options=uf_opts, default=uf_opts if len(uf_opts) <= 5 else [])
    muni_list = sorted(df['no_municipio_prova'].dropna().unique().tolist()) if 'no_municipio_prova' in df.columns else []
    selected_muni = st.sidebar.selectbox("Filtrar por Município (opcional)", options=["(todos)"] + muni_list, index=0)
    plot_var = st.sidebar.selectbox("Variável para histograma / boxplot", options=notas_presentes, index=0 if notas_presentes else None)
    bins = st.sidebar.slider("Número de bins (histograma)", min_value=10, max_value=200, value=60, step=5)

    if plot_var and plot_var in df.columns and df[plot_var].notna().any():
        var_min = float(np.nanmin(df[plot_var])); var_max = float(np.nanmax(df[plot_var]))
    else:
        var_min, var_max = 0.0, 1000.0
    range_slider = st.sidebar.slider(f"Intervalo {plot_var}", min_value=var_min, max_value=var_max, value=(var_min, var_max))
    scatter_x = st.sidebar.selectbox("Scatter X", options=notas_presentes, index=0 if notas_presentes else None)
    scatter_y = st.sidebar.selectbox("Scatter Y", options=[c for c in notas_presentes if c != scatter_x], index=0 if len(notas_presentes) > 1 else 0)
    color_by = st.sidebar.selectbox("Colorir por", options=['sg_uf_prova', 'no_municipio_prova', None] if 'sg_uf_prova' in df.columns else [None], index=0)

    df_f = df.copy()
    if selected_year:
        df_f = df_f[df_f['nu_ano'] == int(selected_year)]
    if selected_ufs:
        df_f = df_f[df_f['sg_uf_prova'].isin(selected_ufs)]
    if selected_muni and selected_muni != "(todos)":
        df_f = df_f[df_f['no_municipio_prova'] == selected_muni]
    if plot_var and plot_var in df_f.columns:
        df_f = df_f[(df_f[plot_var] >= range_slider[0]) & (df_f[plot_var] <= range_slider[1])]

    st.subheader("Resumo rápido dos dados filtrados")
    c1, c2, c3 = st.columns(3)
    c1.metric("Linhas (filtradas)", len(df_f))
    c2.metric("Municípios únicos", int(df_f['no_municipio_prova'].nunique()) if 'no_municipio_prova' in df_f.columns else 0)
    c3.metric("UFs únicas", int(df_f['sg_uf_prova'].nunique()) if 'sg_uf_prova' in df_f.columns else 0)

    st.subheader("Estatísticas descritivas (notas)")
    if notas_presentes:
        stats = summary_stats(df_f, notas_presentes)
        st.dataframe(stats)
    else:
        st.info("Nenhuma nota detectada para estatísticas.")

    st.markdown("---")
    left_col, right_col = st.columns((2,1))
    with left_col:
        st.subheader(f"Histograma — {plot_var}")
        if plot_var and plot_var in df_f.columns:
            fig_hist = px.histogram(df_f, x=plot_var, nbins=bins, title=f"Histograma — {plot_var}", marginal="box",
                                    hover_data=['no_municipio_prova','sg_uf_prova'] if 'no_municipio_prova' in df_f.columns else None)
            fig_hist.update_layout(bargap=0.05)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Variável do histograma não disponível.")

        st.subheader(f"Scatter: {scatter_x} x {scatter_y}")
        hover_cols = ['no_municipio_prova', 'sg_uf_prova'] + notas_presentes
        hover_cols = [c for c in hover_cols if c in df_f.columns]
        if scatter_x and scatter_y and scatter_x in df_f.columns and scatter_y in df_f.columns:
            fig_scat = px.scatter(df_f, x=scatter_x, y=scatter_y, hover_data=hover_cols,
                                 color=color_by if (color_by and color_by in df_f.columns) else None,
                                 title=f"{scatter_y} vs {scatter_x}")
            fig_scat.update_traces(marker=dict(size=7, opacity=0.75))
            st.plotly_chart(fig_scat, use_container_width=True)
        else:
            st.info("Variáveis para scatter não disponíveis.")

        st.subheader("Boxplot por UF (se disponível)")
        if 'sg_uf_prova' in df_f.columns and plot_var in df_f.columns:
            fig_box = px.box(df_f, x='sg_uf_prova', y=plot_var, points='outliers', title=f"Boxplot {plot_var} por UF")
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Boxplot por UF não disponível (falta coluna 'sg_uf_prova' ou variável).")

    with right_col:
        st.subheader("Matriz de correlação (notas)")
        if len(notas_presentes) >= 2:
            corr = df_f[notas_presentes].corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlação entre notas")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Poucas notas para correlação.")

        st.subheader("Mapa (LONGITUDE / LATITUDE)")
        if 'longitude' in df_f.columns and 'latitude' in df_f.columns and df_f[['longitude','latitude']].notna().any().any():
            map_df = df_f.dropna(subset=['longitude','latitude'])
            if len(map_df) > 5000:
                map_df = map_df.sample(5000, random_state=42)
                st.caption("Amostrando 5000 pontos (amostragem por performance).")
            fig_map = px.scatter_mapbox(map_df, lat="latitude", lon="longitude",
                                        hover_name="no_municipio_prova" if 'no_municipio_prova' in map_df.columns else None,
                                        hover_data=[c for c in notas_presentes if c in map_df.columns],
                                        color=plot_var if plot_var in map_df.columns else None,
                                        zoom=3, height=500)
            fig_map.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Longitude/Latitude ausentes — mapa indisponível.")

        st.subheader("Tabela (visualização) — use o cursor/scroll")
        st.dataframe(df_f.head(200))

    st.markdown("---")
    if 'no_municipio_prova' in df_f.columns and notas_presentes:
        st.subheader("Ranking de municípios por média das notas (interativo)")
        agg_muni = df_f.groupby('no_municipio_prova')[notas_presentes].mean().reset_index()
        agg_muni['media_total'] = agg_muni[notas_presentes].mean(axis=1)
        agg_sorted = agg_muni.sort_values('media_total', ascending=False)
        top_k = st.slider("Top K municípios para mostrar", min_value=5, max_value=100, value=10, key="topk_enem")
        st.dataframe(agg_sorted[['no_municipio_prova','media_total'] + notas_presentes].head(top_k))

# -------------------------
# TAB 3 — Modelagem & Previsão
# -------------------------
with tab_model:
    st.header("Modelagem & Previsão — instruções e execução detalhada")
    st.markdown(textwrap.dedent("""
    **Como usar**
    1. Escolha a variável alvo (target) — preferencialmente `nota_media_5_notas` se existir.
    2. Selecione as features (variáveis explicativas) que farão sentido (ex.: outras notas, indicadores por município).
    3. Ajuste o tamanho do teste e a seed (random state).
    4. Clique em **Treinar e comparar modelos**.
    5. Após o treino, o app mostrará as métricas de cada modelo, o eleito como melhor e uma explicação detalhada.
    6. Se um pipeline foi salvo, você poderá **fazer predições interativas** na mesma aba.
    """))

    df_all = st.session_state.get('df_enem', pd.DataFrame())
    if df_all.empty:
        df_disk = load_enem_from_disk(DEFAULT_ENEM)
        if not df_disk.empty:
            st.session_state['df_enem'] = df_disk
            df_all = df_disk

    if df_all.empty:
        st.info("Sem dataset para treinar. Faça upload no topo ou coloque o arquivo em data/raw/ e recarregue.")
        st.stop()

    df_all.columns = [c.lower().strip() for c in df_all.columns]

    # detectar targets com mesma heurística do EDA
    numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
    possible_targets = [c for c in numeric_cols if c in ENEM_NOTAS_LOWER]
    if not possible_targets:
        possible_targets = [c for c in df_all.columns if 'nota' in c]
        possible_targets = [c for c in possible_targets if c in numeric_cols]  # manter numericas
    if not possible_targets:
        possible_targets = numeric_cols[:3]  # fallback mínimo

    if not possible_targets:
        st.info("Nenhuma coluna de nota numérica detectada para modelagem.")
        st.stop()

    target = st.selectbox("Variável alvo (target)", options=possible_targets, index=possible_targets.index('nota_media_5_notas') if 'nota_media_5_notas' in possible_targets else 0)
    default_feats = [c for c in ENEM_NOTAS_LOWER if c in df_all.columns and c != target]
    chosen_feats = st.multiselect("Escolha features (variáveis explicativas)", options=[c for c in df_all.columns if c != target], default=default_feats)

    test_size = st.slider("Tamanho do conjunto de teste (%)", min_value=5, max_value=50, value=20, step=5)
    random_state = st.number_input("Random state (seed)", value=42, step=1)

    if st.button("Treinar e comparar modelos"):
        if not chosen_feats:
            st.error("Escolha ao menos 1 feature.")
        else:
            X = df_all[chosen_feats].copy()
            y = pd.to_numeric(df_all[target], errors='coerce')
            mask = y.notna() & X.notna().all(axis=1)
            X = X.loc[mask]; y = y.loc[mask]
            if len(X) < 10:
                st.error("Dados insuficientes após limpeza para treinar (menos de 10 linhas).")
            else:
                with st.spinner("Treinando modelos..."):
                    results = train_and_evaluate_models(X, y, test_size=test_size/100.0, random_state=int(random_state), model_dir=MODEL_DIR)
                st.success("Treinamento concluído.")

                # tabela de métricas
                res_df = pd.DataFrame(results['metrics']).T.reset_index().rename(columns={'index':'model'})
                st.subheader("Métricas por modelo (no conjunto de teste)")
                st.dataframe(res_df.style.format({"r2": "{:.4f}", "mse": "{:.4f}"}))

                # gráfico barras métricas
                if not res_df.empty and 'r2' in res_df.columns:
                    fig = px.bar(res_df, x='model', y=['r2','mse'], barmode='group', title="Comparação de métricas (R2 e MSE)")
                    st.plotly_chart(fig, use_container_width=True)

                # Melhor modelo e explicação
                best = results.get('best_model')
                st.markdown("### Modelo eleito como o melhor")
                if best:
                    st.markdown(f"- **Melhor modelo (critério):** **{best}**")
                    best_metrics = results['metrics'][best]
                    st.write(f"R² (melhor): {best_metrics['r2']:.4f}, MSE: {best_metrics['mse']:.4f}")
                    st.markdown("**Por que este modelo foi eleito?**")
                    explanation = []
                    sorted_by_r2 = sorted(results['metrics'].items(), key=lambda x: x[1]['r2'], reverse=True)
                    top_r2_name, top_r2_vals = sorted_by_r2[0]
                    if top_r2_name == best:
                        explanation.append(f"- Ele obteve o **maior R²** entre os candidatos ({best_metrics['r2']:.4f}), indicando melhor capacidade de explicar a variância observada no conjunto de teste.")
                    else:
                        explanation.append(f"- A seleção prioriza **R²**; {best} foi escolhido pelo balanço entre R² e MSE (desempate por MSE).")
                    other_ms = {k: v['mse'] for k, v in results['metrics'].items() if k != best}
                    if other_ms:
                        min_other_mse_name = min(other_ms.items(), key=lambda x: x[1])[0]
                        if best_metrics['mse'] <= other_ms[min_other_mse_name]:
                            explanation.append(f"- O MSE do modelo ({best_metrics['mse']:.4f}) é competitivo em relação aos demais.")
                        else:
                            explanation.append(f"- O MSE ({best_metrics['mse']:.4f}) é maior que o menor MSE observado ({min_other_mse_name}), mas priorizamos R² para a escolha do 'melhor'.")
                    explanation.append("- Em resumo: eleito por melhor balanço entre explicabilidade (R²) e erro (MSE) no conjunto de teste.")
                    for line in explanation:
                        st.write(line)
                else:
                    st.info("Não foi possível determinar o melhor modelo automaticamente.")

                st.write("Modelos serializados (pipelines) salvos em:", results.get('model_paths', {}))

    st.markdown("---")
    st.subheader("Previsão Interativa (usar o melhor pipeline salvo)")
    pipeline = load_best_pipeline(MODEL_DIR)
    if pipeline is None:
        st.info("Ainda não há pipeline salvo em models/. Treine um modelo para habilitar predição interativa.")
    else:
        st.success("Pipeline carregado com sucesso.")
        feature_names, feature_defaults = load_feature_metadata(MODEL_DIR)
        if feature_names is None or feature_defaults is None:
            st.error("Metadados de features ausentes. Treine um modelo para gerar defaults (medianas).")
        else:
            st.write("Insira valores para as features (padrões = medianas calculadas durante o treino).")
            user_vals = {}
            for feat in feature_names:
                default_val = float(feature_defaults.get(feat, 0.0)) if feat in feature_defaults else 0.0
                user_vals[feat] = st.number_input(feat, value=default_val, format="%.6f")
            if st.button("Prever"):
                X_input = pd.DataFrame([user_vals], columns=feature_names)
                try:
                    yhat = pipeline.predict(X_input)
                    pred_val = float(yhat[0])
                    st.success(f"Predição para o target selecionado (no último treino): {pred_val:.4f}")
                    st.write("Interpretação: esse valor é a previsão numérica da variável alvo segundo o pipeline treinado.")
                    # Estimativa heurística de UF para a nota prevista:
                    df_ref = st.session_state.get('df_enem', pd.DataFrame()).copy()
                    if 'sg_uf_prova' in df_ref.columns and (target in df_ref.columns or 'nota_media_5_notas' in df_ref.columns):
                        compare_target = 'nota_media_5_notas' if 'nota_media_5_notas' in df_ref.columns else (target if 'target' in locals() else df_ref.select_dtypes(include=[np.number]).columns[0])
                        df_ref[compare_target] = pd.to_numeric(df_ref[compare_target], errors='coerce')
                        uf_means = df_ref.dropna(subset=[compare_target, 'sg_uf_prova']).groupby('sg_uf_prova')[compare_target].mean().reset_index()
                        if not uf_means.empty:
                            uf_means['dist'] = (uf_means[compare_target] - pred_val).abs()
                            uf_sorted = uf_means.sort_values('dist').head(5)
                            st.markdown("**Estimativa heurística: UFs cujas médias de nota estão mais próximas da previsão** (ordem: mais provável → menos provável):")
                            st.dataframe(uf_sorted.rename(columns={compare_target: 'uf_mean', 'sg_uf_prova': 'uf'}).reset_index(drop=True))
                            st.write("Nota: isto é uma suposição simples baseada em médias por UF na sua amostra — não é uma previsão precisa do local do aluno.")
                        else:
                            st.info("Não foi possível calcular médias por UF (poucos dados válidos).")
                    else:
                        st.info("Não há coluna `sg_uf_prova` ou coluna de nota para estimar UFs; impossível fazer a suposição de estado.")
                except Exception as e:
                    st.error(f"Erro ao aplicar o pipeline / modelo: {e}")

# -------------------------
# TAB 4 — Pesquisa Regiões (Análise detalhada)
# -------------------------
with tab_regions:
    st.header("Pesquisa: Regiões mais e menos desenvolvidas — análise prática")
    st.markdown(textwrap.dedent("""
    Esta aba apresenta:
    1. Um resumo conceitual (macro-regiões e desenvolvimento).
    2. Uma análise prática: tomada da variável alvo (preferência por `nota_media_5_notas`) e verificação
       se as 10 maiores, 10 menores e 10 próximas da média vêm de UFs/regiões mais/menos desenvolvidas.
    """))

    st.markdown("**Resumo conceitual (sintético)**: Em termos agregados, Sudeste/Sul e Distrito Federal costumam apresentar maiores níveis de PIB per capita e IDH. Norte e Nordeste, em média, apresentam valores menores. O app usa heurística macro-regional; posso integrar PIB per capita ou IDHM por UF se desejar maior precisão.")

    df_all = st.session_state.get('df_enem', pd.DataFrame())
    if df_all.empty:
        st.info("Sem dataset carregado. Faça upload no topo para executar a análise regional.")
        st.stop()

    df_all.columns = [c.lower().strip() for c in df_all.columns]

    # detectar notas com a mesma heurística do EDA
    notas_presentes = [c for c in ENEM_NOTAS_LOWER if c in df_all.columns]
    if not notas_presentes:
        notas_presentes = [c for c in df_all.columns if 'nota' in c]
    if not notas_presentes:
        st.info("Nenhuma coluna de nota detectada no dataset para análise regional.")
        st.stop()

    target_for_region = 'nota_media_5_notas' if 'nota_media_5_notas' in notas_presentes else notas_presentes[0]
    st.markdown(f"**Variável usada para as comparações:** `{target_for_region}`")

    # criar mapeamento UF -> macro_region e desenv_group
    if 'sg_uf_prova' not in df_all.columns:
        st.info("Coluna `sg_uf_prova` ausente — não é possível agrupar por UF automaticamente.")
        st.stop()

    # mapa de siglas para macro regiões (todas lowercase)
    uf_to_region = {
        'ac':'N','am':'N','ap':'N','pa':'N','ro':'N','rr':'N','to':'N',
        'al':'NE','ba':'NE','ce':'NE','ma':'NE','pb':'NE','pe':'NE','pi':'NE','rn':'NE','se':'NE',
        'es':'SE','mg':'SE','rj':'SE','sp':'SE',
        'pr':'S','rs':'S','sc':'S',
        'df':'CO','go':'CO','mt':'CO','ms':'CO'
    }

    df_all['sg_uf_prova'] = df_all['sg_uf_prova'].astype(str).str.lower().str.strip()
    df_all['macro_region'] = df_all['sg_uf_prova'].map(uf_to_region).fillna('UNK')
    df_all['desenv_group'] = df_all['macro_region'].apply(lambda r: 'mais_desenvolvido' if r in ['se','s','co'] else ('menos_desenvolvido' if r in ['n','ne'] else 'outros'))

    # explicar 'outros' com listagem
    unknown_values = sorted(df_all.loc[df_all['macro_region']=='UNK', 'sg_uf_prova'].dropna().unique().tolist())
    if unknown_values:
        st.warning(f"Atenção: as seguintes siglas/valores não foram reconhecidos e foram categorizados como 'outros' / 'UNK': {unknown_values}")
        st.markdown("**Por que isso acontece?** Pode ser por: grafia diferente (ex: 'SP ' com espaço), nomes completos em vez de sigla (ex: 'Sao Paulo'), valores faltantes ou incorretos. Para corrigir, verifique a coluna `sg_uf_prova` no seu dataset e padronize as siglas (ex: 'sp', 'rj').")

    # garantir coluna numérica do target
    df_all[target_for_region] = pd.to_numeric(df_all[target_for_region], errors='coerce')
    df_valid = df_all.dropna(subset=[target_for_region, 'sg_uf_prova']).copy()
    if df_valid.empty:
        st.info("Após conversão, não há observações válidas com o target e a UF; verifique o dataset.")
        st.stop()

    # 10 maiores / menores / mid10
    top10 = df_valid.sort_values(target_for_region, ascending=False).head(10)
    bottom10 = df_valid.sort_values(target_for_region, ascending=True).head(10)
    median_val = df_valid[target_for_region].median()
    df_valid['dist_med'] = (df_valid[target_for_region] - median_val).abs()
    mid10 = df_valid.sort_values('dist_med').head(10)

    def summary_group(df_slice, name):
        st.subheader(f"{name}")
        st.write(f"Exemplo — mostrando colunas: ['no_municipio_prova', 'sg_uf_prova', '{target_for_region}', 'desenv_group']")
        cols_to_show = [c for c in ['no_municipio_prova', 'sg_uf_prova', target_for_region, 'desenv_group'] if c in df_slice.columns]
        st.dataframe(df_slice[cols_to_show].reset_index(drop=True))
        counts = df_slice['desenv_group'].value_counts().to_dict()
        total = len(df_slice)
        st.write("Distribuição por grupo de desenvolvimento (contagem e %):")
        for grp, cnt in counts.items():
            st.write(f"- {grp}: {cnt} ({cnt/total*100:.1f}%)")
        uf_counts = df_slice['sg_uf_prova'].value_counts().to_dict()
        st.write("UFs presentes (contagem):")
        st.write(uf_counts)
        return counts

    st.markdown("### 10 observações com maiores notas")
    top_counts = summary_group(top10, "Top 10 — maiores notas")

    st.markdown("### 10 observações com menores notas")
    bottom_counts = summary_group(bottom10, "Bottom 10 — menores notas")

    st.markdown("### 10 observações mais próximas da mediana (as '10 médias')")
    mid_counts = summary_group(mid10, "Mid 10 — próximas da média")

    # interpretação automática
    def interpret_counts(counts, label):
        total = sum(counts.values()) if counts else 0
        if total == 0:
            return f"No conjunto {label} não há observações válidas."
        more = counts.get('mais_desenvolvido', 0)
        less = counts.get('menos_desenvolvido', 0)
        outros = counts.get('outros', 0)
        max_group = max([('mais_desenvolvido', more), ('menos_desenvolvido', less), ('outros', outros)], key=lambda x: x[1])
        grp_name, grp_count = max_group
        prop = grp_count / total
        if prop >= 0.6:
            interp = f"A maioria ({grp_count}/{total} = {prop*100:.1f}%) das observações em **{label}** pertence ao grupo **{grp_name}** — isto é uma evidência forte de que {label.lower()} tende a concentrar-se nesse grupo."
        elif prop >= 0.4:
            interp = f"Houve uma tendência moderada: {grp_count}/{total} ({prop*100:.1f}%) das observações em **{label}** pertence ao grupo **{grp_name}**, mas não é uma maioria muito decisiva."
        else:
            interp = f"Não há predominância clara em **{label}**; os registros estão distribuídos entre diferentes grupos."
        return interp

    st.markdown("---")
    st.subheader("Interpretação automática dos três conjuntos")
    st.write("Top 10 (maiores notas): " + interpret_counts(top_counts, "Top 10 — maiores notas"))
    st.write("Bottom 10 (menores notas): " + interpret_counts(bottom_counts, "Bottom 10 — menores notas"))
    st.write("Mid 10 (próximas da média): " + interpret_counts(mid_counts, "Mid 10 — próximas da média"))

    st.markdown("---")
    st.write(textwrap.dedent("""
    **Observações importantes sobre interpretação**
    - Esta análise é **exploratória** e depende estritamente da sua amostra. Amostras não representativas podem distorcer conclusões.
    - 'outros' significa que a sigla de UF não foi reconhecida pelo mapeamento automático (grafia, valores faltantes, nomes em vez de siglas etc.). Verifique sua coluna `sg_uf_prova`.
    - Se quiser que eu refine a classificação usando **PIB per capita (IBGE)** ou **IDHM (Atlas/PNUD)** por UF, eu adiciono rotina para baixar esses indicadores e unir por UF automaticamente.
    - Para testes estatísticos formais (t-test, ANOVA), peça que eu acrescente essa seção.
    """))

# -------------------------
# TAB 5 — Downloads / MLOps
# -------------------------
with tab_downloads:
    st.header("Downloads — arquivos MLOps gerados (data_ingestion, processing, modeling, docs, ZIP)")

    st.markdown("Nesta aba você pode baixar arquivos úteis para transformar este projeto em um pipeline modular e reprodutível. "
                "Os arquivos abaixo são gerados a partir do template usado no projeto. Você pode baixar individualmente ou baixar um ZIP contendo todos.")

    # --- file contents (strings) ---
    data_ingestion_py = textwrap.dedent(r'''
    """
    data_ingestion.py
    - Lê arquivo de data/raw/
    - Valida colunas mínimas
    - Salva cópia processável em data/processed/enem_clean.csv
    """
    import os
    import pandas as pd

    def ingest(input_path=None, output_path="data/processed/enem_clean.csv"):
        if input_path is None:
            input_path = os.path.join("data","raw","Enem_2024_Amostra_Perfeita.xlsx")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")
        df = pd.read_excel(input_path, engine="openpyxl")
        df.columns = [str(c).lower().strip() for c in df.columns]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Ingested: {len(df)} rows -> {output_path}")
        return df

    if __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", help="caminho para o arquivo .xlsx (opcional)")
        parser.add_argument("--output", default="data/processed/enem_clean.csv")
        args = parser.parse_args()
        ingest(args.input, args.output)
    ''')

    data_processing_py = textwrap.dedent(r'''
    """
    data_processing.py
    - Define funções e um objeto ColumnTransformer reutilizável
    - Exporta `build_preprocessor(df, features)` e `prepare_X_y(df, features, target)`
    """
    import pandas as pd
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    import sklearn
    from packaging.version import Version

    def make_onehot():
        from sklearn.preprocessing import OneHotEncoder
        try:
            if Version(sklearn.__version__) >= Version("1.2"):
                return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            else:
                return OneHotEncoder(handle_unknown='ignore', sparse=False)
        except Exception:
            try:
                return OneHotEncoder(handle_unknown='ignore')
            except Exception:
                return OneHotEncoder()

    def build_numeric_transformer():
        return Pipeline([
            ("imputer", SimpleImputer(strategy='median')),
            ("scaler", StandardScaler())
        ])

    def build_preprocessor(df, features):
        num_cols = df[features].select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in features if c not in num_cols]
        preprocessor = ColumnTransformer([
            ("num", build_numeric_transformer(), num_cols),
            ("cat", make_onehot(), cat_cols)
        ], remainder='drop')
        return preprocessor

    def prepare_X_y(df, features, target):
        df2 = df.copy()
        X = df2[features]
        y = pd.to_numeric(df2[target], errors='coerce')
        mask = y.notna() & X.notna().all(axis=1)
        return X.loc[mask], y.loc[mask]

    def detect_notas(df):
        expected = [
            'nota_cn_ciencias_da_natureza','nota_ch_ciencias_humanas','nota_lc_linguagens_e_codigos',
            'nota_mt_matematica','nota_redacao','nota_media_5_notas'
        ]
        notas = [c for c in expected if c in df.columns]
        if notas:
            return notas
        notas = [c for c in df.columns if 'nota' in c]
        if notas:
            return notas
        return []
    ''')

    modeling_py = textwrap.dedent(r'''
    """
    modeling.py
    - Treina modelos e serializa pipelines
    - Funções: train_models(X,y, preprocessor, model_dir), load_best_pipeline(model_dir)
    """
    import os
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error

    def train_models(X, y, preprocessor, model_dir="models", test_size=0.2, random_state=42):
        os.makedirs(model_dir, exist_ok=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=200, random_state=random_state),
            'LinearRegression': LinearRegression(),
            'SVR': SVR()
        }
        results = {}
        model_paths = {}
        for name, model in models.items():
            pipe = Pipeline([('preproc', preprocessor), ('model', model)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            r2 = r2_score(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            results[name] = {'r2': float(r2), 'mse': float(mse)}
            path = os.path.join(model_dir, f'pipeline_{name}.joblib')
            joblib.dump(pipe, path)
            model_paths[name] = path
        sorted_models = sorted(results.items(), key=lambda x: (-x[1]['r2'], x[1]['mse']))
        best_name = sorted_models[0][0]
        best_path = model_paths[best_name]
        joblib.dump(joblib.load(best_path), os.path.join(model_dir, 'best_pipeline.joblib'))
        joblib.dump(X.columns.tolist(), os.path.join(model_dir, 'feature_names.joblib'))
        joblib.dump(X.median().to_dict(), os.path.join(model_dir, 'feature_defaults.joblib'))
        return {'results': results, 'best': best_name, 'paths': model_paths}

    def load_best_pipeline(model_dir='models'):
        path = os.path.join(model_dir, 'best_pipeline.joblib')
        if os.path.exists(path):
            return joblib.load(path)
        return None
    ''')

    evaluate_py = textwrap.dedent(r'''
    """
    evaluate.py
    - Funções para exibir e salvar métricas, gerar gráfico de feature importance
    """
    import os
    import joblib
    import pandas as pd
    import matplotlib.pyplot as plt

    def load_metrics_from_paths(model_paths):
        metrics = {}
        for name, path in model_paths.items():
            pipe = joblib.load(path)
            try:
                model = pipe.named_steps['model']
                if hasattr(model, 'feature_importances_'):
                    metrics[name] = {'feature_importances': model.feature_importances_}
            except Exception:
                pass
        return metrics

    def save_metrics_table(results, out_csv='models/metrics_summary.csv'):
        rows = []
        for m, vals in results.items():
            rows.append({'model': m, 'r2': vals['r2'], 'mse': vals['mse']})
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        return df
    ''')

    database_doc_md = textwrap.dedent(r'''
    # database_doc.md

    ## Nome do dataset
    - Enem_2024_Amostra_Perfeita.xlsx (coloque a fonte/link do dataiesb.com se aplicável)

    ## Contexto do Negócio
    - Objetivo: analisar relação entre notas do ENEM e indicadores regionais; treinar modelos para prever nota e investigar se notas altas concentram-se em regiões mais desenvolvidas.

    ## Modelo Conceitual
    - Tabela única (cada linha = observação por participante/município). Não há joins necessários na versão atual.

    ## Dicionário de Dados (exemplo)
    | Coluna | Tipo | Descrição | Exemplo/Valores válidos |
    |---|---|---:|---|
    | sg_uf_prova | object | Sigla da unidade federativa (UF) | 'sp', 'rj', 'df' |
    | no_municipio_prova | object | Nome do município | 'Brasilia' |
    | nota_mt_matematica | float | Nota de matemática | 560.12 |
    | nota_redacao | float | Nota da redação | 700.0 |
    | longitude | float | Longitude (opcional) | -47.8825 |
    | latitude | float | Latitude (opcional) | -15.7941 |

    ## Pré-Processamento
    - Normalização de nomes de colunas (lower + strip)
    - Conversão numérica de notas (coerção com `pd.to_numeric(errors='coerce')`)
    - Tratamento de nulos: `SimpleImputer(strategy='median')` para numéricos
    - Categóricas: `OneHotEncoder(handle_unknown='ignore')`
    - Escalonamento: `StandardScaler` para variáveis numéricas
    ''')

    requirements_txt = textwrap.dedent(r'''
    streamlit
    pandas
    numpy
    plotly
    openpyxl
    scikit-learn
    joblib
    matplotlib
    packaging
    ''')

    readme_md = textwrap.dedent(r'''
    # P2 — Projeto MLOps (ENEM 2024)

    ## Estrutura de pastas sugerida
    ```
    data/
      raw/                # xlsx original
      processed/          # csv limpo
    models/               # pipelines salvos
    src/                  # scripts (data_ingestion, data_processing, modeling, evaluate)
    app_streamlit.py
    requirements.txt
    README.md
    ```

    ## Como rodar
    1. Instale dependências: `pip install -r requirements.txt`
    2. Ingestão: `python data_ingestion.py --input data/raw/Enem_2024_Amostra_Perfeita.xlsx`
    3. Treino: rode os scripts `data_processing.py` + `modeling.py` (ou use o Streamlit)
    4. Dashboard: `streamlit run app_streamlit.py`

    ## Observações
    - Se houver warnings de versão do scikit-learn, recomendo treinar os modelos no mesmo ambiente em que irá rodar o app.
    ''')

    run_all_sh = textwrap.dedent(r'''
    #!/usr/bin/env bash
    python data_ingestion.py
    python - <<'PY'
    from data_processing import detect_notas, prepare_X_y, build_preprocessor
    import pandas as pd
    from modeling import train_models
    df = pd.read_csv('data/processed/enem_clean.csv')
    notas = detect_notas(df)
    if not notas:
        raise SystemExit('Nenhuma nota detectada')
    target = 'nota_media_5_notas' if 'nota_media_5_notas' in notas else notas[0]
    features = [c for c in df.columns if c!=target][:10]
    X,y = prepare_X_y(df, features, target)
    preproc = build_preprocessor(df, features)
    train_models(X,y,preproc)
    PY
    streamlit run app_streamlit.py
    ''')

    app_modular_py = textwrap.dedent(r'''
    # app_streamlit_modular.py (exemplo de app que importa os módulos)
    import os
    import textwrap
    import streamlit as st
    import pandas as pd
    from data_processing import detect_notas, prepare_X_y, build_preprocessor
    from modeling import train_models, load_best_pipeline

    st.set_page_config(page_title='ENEM 2024 — Modular App', layout='wide')

    DEFAULT_PROCESSED = os.path.join('data','processed','enem_clean.csv')
    uploaded = st.file_uploader('Upload do arquivo .xlsx (opcional)', type=['xlsx'])
    if uploaded is not None:
        df = pd.read_excel(uploaded, engine='openpyxl')
        df.columns = [str(c).lower().strip() for c in df.columns]
        df.to_csv(DEFAULT_PROCESSED, index=False)
        st.success('Arquivo processado e salvo em data/processed/')
    else:
        if os.path.exists(DEFAULT_PROCESSED):
            df = pd.read_csv(DEFAULT_PROCESSED)
        else:
            st.warning('Nenhum arquivo encontrado. Faça upload ou coloque o xlsx em data/raw/')
            st.stop()

    notas = detect_notas(df)
    st.write('Notas detectadas:', notas)
    ''')

    # mapping of display name to content and file name
    files = [
        ("data_ingestion.py", data_ingestion_py),
        ("data_processing.py", data_processing_py),
        ("modeling.py", modeling_py),
        ("evaluate.py", evaluate_py),
        ("database_doc.md", database_doc_md),
        ("requirements.txt", requirements_txt),
        ("README.md", readme_md),
        ("run_all.sh", run_all_sh),
        ("app_streamlit_modular.py", app_modular_py)
    ]

    # show download buttons for each file
    st.subheader("Arquivos individuais")
    for fname, content in files:
        st.download_button(label=f"Baixar {fname}", data=content, file_name=fname, mime="text/plain")

    st.markdown("---")
    st.subheader("Baixar todos os arquivos como ZIP")
    # build zip in-memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, content in files:
            zf.writestr(fname, content)
    zip_buffer.seek(0)
    st.download_button(label="Baixar ZIP com todos os arquivos (.zip)", data=zip_buffer.getvalue(),
                       file_name="p2_mlops_project_files.zip", mime="application/zip")

    st.markdown("---")
    st.write("Instruções rápidas:")
    st.write("- Coloque seu `Enem_2024_Amostra_Perfeita.xlsx` em `data/raw/` ou use o uploader na aba principal.")
    st.write("- Depois de baixar os arquivos, ajuste `data_ingestion.py` se o nome do arquivo for diferente.")
    st.write("- Para treinar localmente, execute `python data_ingestion.py` e depois rode os scripts `data_processing.py`/`modeling.py` ou use a aba Modelagem no Streamlit.")
    st.write("Se quiser, eu já posso gerar o ZIP com versões ajustadas ao seu dataset (preciso de 5–10 linhas do CSV para ajustar automaticamente os nomes das colunas).")

# FIM do app