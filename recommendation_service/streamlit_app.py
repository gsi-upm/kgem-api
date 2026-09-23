import os

import pandas as pd
import requests
import streamlit as st


API_BASE = os.getenv("RECOMMENDATION_API_BASE", "http://localhost:8002").rstrip("/")

ENTITY_NAMES = {
    "brazil": "Brasil",
    "burma": "Birmania",
    "china": "China",
    "cuba": "Cuba",
    "egypt": "Egipto",
    "india": "India",
    "indonesia": "Indonesia",
    "israel": "Israel",
    "jordan": "Jordania",
    "netherlands": "Países Bajos",
    "poland": "Polonia",
    "uk": "Reino Unido",
    "usa": "Estados Unidos",
    "ussr": "URSS",
}

ARTICLES = {
    "atlantic": {
        "title": "Estados Unidos y Reino Unido refuerzan su alianza",
        "summary": "Ambos países anuncian una nueva agenda de cooperación diplomática.",
        "entities": ["usa", "uk"],
    },
    "trade": {
        "title": "Países Bajos negocia un acuerdo con Estados Unidos",
        "summary": "La reunión se centra en comercio, innovación y relaciones bilaterales.",
        "entities": ["netherlands", "usa"],
    },
    "asia": {
        "title": "China e India retoman sus conversaciones comerciales",
        "summary": "Las delegaciones buscan abrir una nueva etapa de cooperación regional.",
        "entities": ["china", "india"],
    },
    "middle_east": {
        "title": "Egipto y Jordania coordinan una iniciativa regional",
        "summary": "Los dos gobiernos preparan un programa conjunto de ayuda humanitaria.",
        "entities": ["egypt", "jordan"],
    },
    "forests": {
        "title": "Brasil e Indonesia cooperarán para proteger sus bosques",
        "summary": "El pacto propone compartir tecnología y políticas de conservación.",
        "entities": ["brazil", "indonesia"],
    },
    "europe": {
        "title": "Polonia y Reino Unido amplían su cooperación",
        "summary": "El encuentro aborda seguridad, industria y relaciones europeas.",
        "entities": ["poland", "uk"],
    },
}

FEATURE_NAMES = {
    "mean": "Similitud media",
    "max": "Similitud máxima",
    "median": "Similitud mediana",
    "min": "Similitud mínima",
    "sum": "Similitud acumulada",
    "centroid-Rmean": "Solapamiento (radio medio)",
    "centroid-Rmedian": "Solapamiento (radio mediano)",
    "centroid-Rmax": "Solapamiento (radio máximo)",
    "geometric": "Solapamiento geométrico",
    "distance_centroids": "Distancia entre centroides",
    "distance_geometric": "Distancia entre medianas geométricas",
}


def request_recommendation(payload):
    try:
        response = requests.post(f"{API_BASE}/recommend/", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError as exc:
        raise RuntimeError(
            "No se puede conectar con el recomendador. Arranca la demo con ./demo.sh."
        ) from exc
    except requests.RequestException as exc:
        detail = response.text if "response" in locals() else str(exc)
        try:
            detail = response.json().get("detail", detail)
        except (ValueError, AttributeError):
            pass
        raise RuntimeError(f"El recomendador devolvió un error: {detail}") from exc


def entity_label(entity):
    return f"{ENTITY_NAMES[entity]} · {entity}"


def article_picker(column, heading, widget_key, default_index):
    with column:
        st.markdown(f"#### {heading}")
        article_key = st.selectbox(
            "Selecciona un titular",
            ARTICLES,
            index=default_index,
            format_func=lambda key: ARTICLES[key]["title"],
            key=f"headline-{widget_key}",
        )
        article = ARTICLES[article_key]
        with st.container(border=True):
            st.caption("TITULAR FICTICIO")
            st.markdown(f"### {article['title']}")
            st.write(article["summary"])
        chips = "".join(
            f'<span class="entity-chip">{entity_label(entity)}</span>'
            for entity in article["entities"]
        )
        st.markdown(
            f'<div class="entity-label">ENTIDADES DETECTADAS AUTOMÁTICAMENTE</div>'
            f'<div class="entity-list">{chips}</div>',
            unsafe_allow_html=True,
        )
        return article["entities"]


def show_result(result):
    score = float(result["relevance_score"])
    is_regression = result["mode"] == "regression"
    recommend = score >= 0.5
    verdict = "Sí, recomendar" if recommend else "No recomendar"
    explanation = (
        f"Afinidad estimada: {score:.0%}"
        if is_regression
        else f"Clase predicha: {int(score)} · {'buena' if recommend else 'mala'} recomendación"
    )
    css_class = "positive" if recommend else "negative"

    st.markdown("### Resultado")
    st.markdown(
        f"""
        <div class="result {css_class}">
            <div class="result-label">DECISIÓN DEL RECOMENDADOR</div>
            <div class="result-title">{verdict}</div>
            <div>{explanation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if is_regression:
        st.progress(max(0.0, min(score, 1.0)), text=f"Puntuación original: {score:.3f}")
        st.caption(
            "La API devuelve una puntuación continua. La demo usa 0,50 como umbral visual "
            "para traducirla a una decisión."
        )

    features = result["features"]
    columns = st.columns(3)
    columns[0].metric("Similitud media", f"{features['mean']:.3f}")
    columns[1].metric("Similitud máxima", f"{features['max']:.3f}")
    columns[2].metric("Distancia entre grupos", f"{features['distance_centroids']:.3f}")

    used = result["used_entities"]
    st.caption(
        "Entidades usadas por KGEM-API · A: "
        f"{', '.join(map(entity_label, used['cluster 1']))} · B: "
        f"{', '.join(map(entity_label, used['cluster 2']))}"
    )

    with st.expander("Ver todas las señales y la respuesta técnica"):
        feature_table = pd.DataFrame(
            {
                "Señal": [FEATURE_NAMES.get(name, name) for name in features],
                "Valor": [value for value in features.values()],
            }
        )
        st.dataframe(feature_table, hide_index=True, width="stretch")
        st.json(result)


st.set_page_config(
    page_title="Demo del recomendador · KGEM",
    page_icon="🧭",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp { background: linear-gradient(145deg, #f8fafc 0%, #eef6ff 100%); }
        .block-container { max-width: 1120px; padding-top: 2.5rem; }
        .hero {
            padding: 2.1rem 2.3rem;
            border-radius: 24px;
            background: linear-gradient(120deg, #102a43 0%, #176b87 100%);
            color: white;
            box-shadow: 0 18px 45px rgba(16, 42, 67, .16);
            margin-bottom: 1.5rem;
        }
        .hero h1 { color: white; margin: 0 0 .45rem; font-size: 2.4rem; }
        .hero p { margin: 0; color: #d9f3f3; font-size: 1.08rem; max-width: 760px; }
        .eyebrow { color: #73e0cc; font-weight: 800; letter-spacing: .12em; font-size: .78rem; }
        .flow {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: .75rem;
            flex-wrap: wrap;
            margin: 1.2rem 0 1.7rem;
            color: #334e68;
        }
        .flow span { background: white; border: 1px solid #d9e2ec; padding: .7rem 1rem; border-radius: 999px; }
        .flow b { color: #147d73; }
        .entity-label { color: #627d98; font-size: .72rem; font-weight: 800; letter-spacing: .06em; margin: 1rem 0 .45rem; }
        .entity-list { display: flex; flex-wrap: wrap; gap: .45rem; }
        .entity-chip { background: #d5f5f1; border: 1px solid #8bd8cf; border-radius: 999px; color: #075e54; padding: .35rem .7rem; font-size: .85rem; font-weight: 650; }
        .result {
            padding: 1.4rem 1.6rem;
            border-radius: 18px;
            margin: .5rem 0 1rem;
            border-left: 7px solid;
        }
        .result.positive { background: #e6fcf5; border-color: #0ca678; color: #065f46; }
        .result.negative { background: #fff4e6; border-color: #f08c00; color: #8a4b08; }
        .result-label { font-size: .75rem; font-weight: 800; letter-spacing: .12em; }
        .result-title { font-size: 2rem; font-weight: 800; line-height: 1.25; margin: .2rem 0; }
        div.stButton > button { min-height: 3.1rem; font-size: 1.05rem; font-weight: 700; }
        @media (max-width: 700px) { .hero h1 { font-size: 1.85rem; } }
    </style>
    <section class="hero">
        <div class="eyebrow">KGEM · RECOMENDACIÓN EXPLICABLE</div>
        <h1>¿Encajan estas dos noticias?</h1>
        <p>Compara las entidades de una noticia abierta con las de otra candidata y descubre
        si el modelo la recomendaría.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Esta demo no lee los titulares: simula que un extractor previo ya identificó sus "
    "entidades. KGEM-API compara sus embeddings y el recomendador usa esas señales."
)

left, right = st.columns(2, gap="large")
entities_a = article_picker(left, "1 · Noticia que estás leyendo", "a", 0)
entities_b = article_picker(right, "2 · Noticia candidata", "b", 1)

st.markdown(
    """
    <div class="flow">
        <span>📰 Titulares</span><b>→</b>
        <span>🏷️ Entidades</span><b>→</b>
        <span>🧠 Embeddings KGEM</span><b>→</b>
        <span>🎯 Recomendación</span>
    </div>
    """,
    unsafe_allow_html=True,
)

mode_label = st.radio(
    "¿Qué salida quieres ver?",
    ("Puntuación continua", "Decisión binaria"),
    horizontal=True,
    help="Regresión devuelve una afinidad; clasificación devuelve 0 o 1.",
)
mode = "regression" if mode_label == "Puntuación continua" else "classification"

payload = {
    "news1_entities": entities_a,
    "news2_entities": entities_b,
    "mode": mode,
    "model_name": "random_forest",
    "graph": "nations",
    "embedding_model": "transe",
}

if st.button(
    "Comparar y decidir",
    type="primary",
    width="stretch",
    disabled=not entities_a or not entities_b,
):
    try:
        with st.spinner("KGEM-API está comparando los grupos de entidades…"):
            st.session_state["recommendation"] = (payload, request_recommendation(payload))
    except RuntimeError as exc:
        st.session_state.pop("recommendation", None)
        st.error(str(exc))

stored = st.session_state.get("recommendation")
if stored and stored[0] == payload:
    show_result(stored[1])

st.divider()
st.markdown(
    """
    **Configuración fija de la demo:**

    - Grafo: **Nations**
    - Embeddings: **TransE**
    - Modelo: **Random Forest**

    Los titulares son ficticios y sirven únicamente para hacer visible el flujo.
    """
)
