"""Dashboard de viabilidad para una comunidad energetica del Tolima."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Comunidad Energética | EAFIT 2026",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def validar_acceso() -> None:
    """Bloquea el dashboard hasta que se ingrese la clave solicitada."""
    if "autenticado" not in st.session_state:
        st.session_state.autenticado = False

    if st.session_state.autenticado:
        return

    st.title("🔐 Acceso al dashboard")
    st.write("Ingresa la clave para consultar el análisis de la comunidad energética.")
    clave = st.text_input("Clave", type="password", placeholder="Clave de acceso")
    if st.button("Ingresar", type="primary", use_container_width=True):
        if clave == "1123":
            st.session_state.autenticado = True
            st.rerun()
        else:
            st.error("Clave incorrecta.")
    st.stop()


@st.cache_data(show_spinner=False)
def generar_datos(seed: int = 1123, n_registros: int = 500) -> pd.DataFrame:
    """Crea una serie horaria sintetica de clima, oferta y demanda energetica."""
    rng = np.random.default_rng(seed)
    fecha_hora = pd.date_range("2026-07-01", periods=n_registros, freq="h")
    hora = fecha_hora.hour.to_numpy()
    dia_semana = fecha_hora.dayofweek.to_numpy()

    # Variables meteorologicas correlacionadas con las fuentes renovables.
    perfil_solar = np.maximum(0, np.sin(np.pi * (hora - 6) / 12))
    nubes = np.clip(rng.beta(4.5, 2.2, n_registros), 0.10, 1.0)
    irradiancia = np.clip(1_020 * perfil_solar * nubes + rng.normal(0, 18, n_registros), 0, 1_100)
    velocidad_viento = np.clip(
        4.2 + 1.8 * np.sin(2 * np.pi * (hora + 2) / 24) + rng.normal(0, 1.35, n_registros),
        0,
        13,
    )
    evento_lluvia = rng.random(n_registros) < 0.22
    precipitacion = np.where(evento_lluvia, rng.gamma(1.8, 3.2, n_registros), 0.0)

    # Energia horaria disponible por tecnologia (kWh).
    generacion_solar = np.clip(780 * irradiancia / 1_000 + rng.normal(0, 12, n_registros), 0, 800)
    curva_eolica = np.where(
        velocidad_viento < 3,
        0,
        np.minimum(620, 620 * ((velocidad_viento - 3) / 7) ** 3),
    )
    generacion_eolica = np.clip(curva_eolica + rng.normal(0, 10, n_registros), 0, 620)
    caudal_relativo = pd.Series(precipitacion).rolling(36, min_periods=1).mean().to_numpy()
    generacion_pch = np.clip(180 + 22 * caudal_relativo + rng.normal(0, 9, n_registros), 140, 420)

    # Perfil de demanda: picos residencial matutino y nocturno, y actividad laboral.
    pico_manana = 150 * np.exp(-0.5 * ((hora - 7) / 2.0) ** 2)
    pico_noche = 300 * np.exp(-0.5 * ((hora - 19) / 2.4) ** 2)
    actividad_diurna = np.where((hora >= 8) & (hora <= 17) & (dia_semana < 5), 130, 0)
    demanda = np.clip(430 + pico_manana + pico_noche + actividad_diurna + rng.normal(0, 35, n_registros), 300, 980)

    generacion_total = generacion_solar + generacion_eolica + generacion_pch
    balance = generacion_total - demanda
    decision = np.select(
        [balance > 100, balance >= 0],
        ["Vender a la red", "Cubrir demanda"],
        default="Comprar a la red",
    )

    # Exactamente 500 registros y 10 columnas de tipos mixtos.
    return pd.DataFrame(
        {
            "fecha_hora": fecha_hora,
            "irradiancia_w_m2": irradiancia.round(2),
            "velocidad_viento_m_s": velocidad_viento.round(2),
            "precipitacion_mm": precipitacion.round(2),
            "generacion_solar_kwh": generacion_solar.round(2),
            "generacion_eolica_kwh": generacion_eolica.round(2),
            "generacion_pch_kwh": generacion_pch.round(2),
            "demanda_kwh": demanda.round(2),
            "balance_kwh": balance.round(2),
            "decision": pd.Categorical(
                decision,
                categories=["Vender a la red", "Cubrir demanda", "Comprar a la red"],
            ),
        }
    )


def numero(valor: float, decimales: int = 1) -> str:
    """Da formato numerico en convencion colombiana."""
    return f"{valor:,.{decimales}f}".replace(",", "X").replace(".", ",").replace("X", ".")


with st.sidebar:
    st.markdown("## EAFIT 2026")
    st.markdown("**Ciencia de datos**")
    st.markdown("Profesor Jorge")
    st.markdown("17 de julio")
    st.divider()

validar_acceso()

with st.sidebar:
    st.header("Configuración")
    semilla = st.number_input("Semilla de simulación", min_value=0, max_value=99_999, value=1123)
    if st.button("Regenerar serie", use_container_width=True):
        st.cache_data.clear()

datos = generar_datos(int(semilla))

with st.sidebar:
    st.subheader("Filtros")
    fecha_min = datos["fecha_hora"].min().date()
    fecha_max = datos["fecha_hora"].max().date()
    rango = st.date_input(
        "Periodo",
        value=(fecha_min, fecha_max),
        min_value=fecha_min,
        max_value=fecha_max,
    )
    decisiones_disponibles = datos["decision"].astype(str).unique().tolist()
    decisiones = st.multiselect(
        "Decisión energética",
        decisiones_disponibles,
        default=decisiones_disponibles,
    )

    st.subheader("Reglas de viabilidad")
    reserva = st.slider(
        "Reserva antes de vender (kWh)",
        min_value=0,
        max_value=400,
        value=100,
        step=10,
        help="Solo se considera vendible el excedente que supera esta reserva.",
    )
    precio_venta = st.number_input("Precio de venta ($/kWh)", 0.0, 2_000.0, 420.0, 10.0)
    precio_compra = st.number_input("Precio de compra ($/kWh)", 0.0, 2_000.0, 760.0, 10.0)

    st.subheader("Apariencia")
    tema = st.selectbox("Tema Plotly", ["plotly_white", "plotly_dark", "ggplot2", "seaborn"])
    paleta = st.selectbox("Paleta", ["Viridis", "Turbo", "Plasma", "Cividis", "Blues"])
    if st.button("Cerrar sesión", use_container_width=True):
        st.session_state.autenticado = False
        st.rerun()

inicio, fin = fecha_min, fecha_max
if isinstance(rango, (tuple, list)) and len(rango) == 2:
    inicio, fin = rango
elif isinstance(rango, date):
    inicio = fin = rango

mascara = datos["fecha_hora"].dt.date.between(inicio, fin) & datos["decision"].astype(str).isin(decisiones)
filtrados = datos.loc[mascara].copy()

st.title("⚡ Comunidad energética: oferta, demanda y mercado")
st.caption(
    "Serie sintética horaria · Generación solar, eólica y PCH · Evaluación de compra y venta a la red"
)

if filtrados.empty:
    st.warning("No hay registros para los filtros seleccionados.")
    st.stop()

generacion_total = filtrados[
    ["generacion_solar_kwh", "generacion_eolica_kwh", "generacion_pch_kwh"]
].sum(axis=1)
excedente_vendible = (filtrados["balance_kwh"] - reserva).clip(lower=0)
deficit = (-filtrados["balance_kwh"]).clip(lower=0)
ingreso_estimado = excedente_vendible.sum() * precio_venta
costo_estimado = deficit.sum() * precio_compra
cobertura = generacion_total.sum() / filtrados["demanda_kwh"].sum() * 100
horas_venta = int((excedente_vendible > 0).sum())

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Cobertura renovable", f"{numero(cobertura)} %")
m2.metric("Generación", f"{numero(generacion_total.sum() / 1_000)} MWh")
m3.metric("Demanda", f"{numero(filtrados['demanda_kwh'].sum() / 1_000)} MWh")
m4.metric("Horas viables de venta", horas_venta, delta=f"{horas_venta / len(filtrados):.1%}")
m5.metric("Resultado de mercado", f"${numero(ingreso_estimado - costo_estimado, 0)}")

tab_balance, tab_explorador, tab_viabilidad, tab_estadistica, tab_datos = st.tabs(
    ["Balance energético", "Análisis gráfico", "Viabilidad", "Estadística", "Datos"]
)

numericas = [
    "irradiancia_w_m2",
    "velocidad_viento_m_s",
    "precipitacion_mm",
    "generacion_solar_kwh",
    "generacion_eolica_kwh",
    "generacion_pch_kwh",
    "demanda_kwh",
    "balance_kwh",
]

with tab_balance:
    st.subheader("Serie de tiempo: generación frente a demanda")
    serie_plot = filtrados[["fecha_hora", "demanda_kwh"]].copy()
    serie_plot["generacion_total_kwh"] = generacion_total
    serie_larga = serie_plot.melt(
        id_vars="fecha_hora",
        value_vars=["generacion_total_kwh", "demanda_kwh"],
        var_name="variable",
        value_name="energia_kwh",
    )
    fig_serie = px.line(
        serie_larga,
        x="fecha_hora",
        y="energia_kwh",
        color="variable",
        labels={"fecha_hora": "Fecha y hora", "energia_kwh": "Energía (kWh)", "variable": "Serie"},
        color_discrete_map={"generacion_total_kwh": "#1B9E77", "demanda_kwh": "#D95F02"},
    )
    fig_serie.update_layout(template=tema, hovermode="x unified", height=500)
    st.plotly_chart(fig_serie, use_container_width=True)

    c1, c2 = st.columns(2)
    mezcla = pd.DataFrame(
        {
            "Fuente": ["Solar", "Eólica", "PCH"],
            "Energía (kWh)": [
                filtrados["generacion_solar_kwh"].sum(),
                filtrados["generacion_eolica_kwh"].sum(),
                filtrados["generacion_pch_kwh"].sum(),
            ],
        }
    )
    fig_mezcla = px.pie(
        mezcla,
        names="Fuente",
        values="Energía (kWh)",
        hole=0.5,
        title="Mezcla de generación",
        color_discrete_sequence=["#F4C430", "#74B9FF", "#00A896"],
    )
    fig_mezcla.update_layout(template=tema)
    c1.plotly_chart(fig_mezcla, use_container_width=True)

    frecuencia = filtrados["decision"].astype(str).value_counts().rename_axis("Decisión").reset_index(name="Horas")
    fig_decision = px.bar(
        frecuencia,
        x="Decisión",
        y="Horas",
        color="Decisión",
        title="Decisión operativa por hora",
        color_discrete_map={
            "Vender a la red": "#2A9D8F",
            "Cubrir demanda": "#E9C46A",
            "Comprar a la red": "#E76F51",
        },
    )
    fig_decision.update_layout(template=tema, showlegend=False)
    c2.plotly_chart(fig_decision, use_container_width=True)

with tab_explorador:
    st.subheader("Gráfico dinámico de variables")
    a, b, c = st.columns(3)
    eje_x = a.selectbox("Eje X", ["fecha_hora", *numericas])
    eje_y = b.selectbox("Eje Y", numericas, index=6)
    tipo = c.selectbox("Tipo", ["Línea", "Dispersión", "Barras", "Histograma", "Caja"])
    colorear = st.checkbox("Colorear por decisión", value=tipo in {"Dispersión", "Caja"})
    color = "decision" if colorear else None

    if tipo == "Línea":
        fig = px.line(filtrados.sort_values(eje_x), x=eje_x, y=eje_y, color=color)
    elif tipo == "Dispersión":
        fig = px.scatter(
            filtrados,
            x=eje_x,
            y=eje_y,
            color=color,
            hover_data=["fecha_hora", "decision"],
            opacity=0.75,
        )
    elif tipo == "Barras":
        frecuencia_barra = st.selectbox("Agrupar por tiempo", ["Día", "Semana"])
        regla = "D" if frecuencia_barra == "Día" else "W"
        agregado = (
            filtrados.assign(periodo=filtrados["fecha_hora"].dt.to_period(regla).dt.start_time)
            .groupby("periodo", observed=True)[eje_y]
            .mean()
            .reset_index()
        )
        fig = px.bar(agregado, x="periodo", y=eje_y, color=eje_y, color_continuous_scale=paleta)
    elif tipo == "Histograma":
        intervalos = st.slider("Intervalos", 10, 70, 30)
        fig = px.histogram(filtrados, x=eje_y, color=color, nbins=intervalos, marginal="box")
    else:
        fig = px.box(filtrados, x="decision", y=eje_y, color="decision", points="outliers")

    fig.update_layout(template=tema, height=540, legend_title_text="Decisión")
    st.plotly_chart(fig, use_container_width=True)

    correlacion = filtrados[numericas].corr()
    fig_corr = go.Figure(
        go.Heatmap(
            z=correlacion.values,
            x=correlacion.columns,
            y=correlacion.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=np.round(correlacion.values, 2),
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig_corr.update_layout(title="Correlación entre clima, generación y demanda", template=tema, height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

with tab_viabilidad:
    st.subheader("Escenario económico según demanda y excedentes")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Energía vendible", f"{numero(excedente_vendible.sum() / 1_000)} MWh")
    v2.metric("Energía a comprar", f"{numero(deficit.sum() / 1_000)} MWh")
    v3.metric("Ingresos potenciales", f"${numero(ingreso_estimado, 0)}")
    v4.metric("Costo de compra", f"${numero(costo_estimado, 0)}")

    escenario = filtrados[["fecha_hora", "balance_kwh"]].copy()
    escenario["excedente_vendible_kwh"] = excedente_vendible
    escenario["deficit_kwh"] = deficit
    fig_viabilidad = px.bar(
        escenario,
        x="fecha_hora",
        y=["excedente_vendible_kwh", "deficit_kwh"],
        barmode="group",
        labels={"value": "Energía (kWh)", "variable": "Condición", "fecha_hora": "Fecha"},
        color_discrete_sequence=["#2A9D8F", "#E76F51"],
    )
    fig_viabilidad.add_hline(
        y=reserva,
        line_dash="dash",
        annotation_text="Reserva configurada",
        line_color="#264653",
    )
    fig_viabilidad.update_layout(template=tema, hovermode="x unified", height=500)
    st.plotly_chart(fig_viabilidad, use_container_width=True)

    resultado = ingreso_estimado - costo_estimado
    if resultado > 0:
        st.success(f"El escenario es favorable: el resultado neto estimado es ${numero(resultado, 0)}.")
    else:
        st.warning(
            f"El escenario requiere compras netas por ${numero(abs(resultado), 0)}. "
            "Conviene revisar almacenamiento, gestión de demanda o capacidad instalada."
        )

with tab_estadistica:
    st.subheader("Estadística cuantitativa")
    variable = st.selectbox("Variable cuantitativa", numericas, index=6, key="variable_estadistica")
    serie = filtrados[variable]
    resumen = pd.DataFrame(
        {
            "Métrica": ["Conteo", "Media", "Mediana", "Desv. estándar", "Mínimo", "Q1", "Q3", "Máximo"],
            "Valor": [
                serie.count(), serie.mean(), serie.median(), serie.std(), serie.min(),
                serie.quantile(0.25), serie.quantile(0.75), serie.max(),
            ],
        }
    )
    e1, e2 = st.columns([1, 2])
    e1.dataframe(resumen.style.format({"Valor": "{:,.2f}"}), hide_index=True, use_container_width=True)
    fig_dist = px.histogram(
        filtrados,
        x=variable,
        color="decision",
        marginal="violin",
        nbins=30,
        title=f"Distribución de {variable}",
    )
    fig_dist.update_layout(template=tema)
    e2.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("Estadística cualitativa")
    cualitativa = filtrados["decision"].astype(str).value_counts().rename("Frecuencia").to_frame()
    cualitativa["Porcentaje"] = cualitativa["Frecuencia"] / len(filtrados) * 100
    cualitativa.loc["Total"] = [len(filtrados), 100]
    st.dataframe(cualitativa.style.format({"Porcentaje": "{:.2f}%"}), use_container_width=True)

with tab_datos:
    st.subheader("Datos sintéticos: 500 registros × 10 columnas")
    st.dataframe(
        filtrados,
        hide_index=True,
        use_container_width=True,
        column_config={
            "fecha_hora": st.column_config.DatetimeColumn("Fecha y hora", format="DD/MM/YYYY HH:mm"),
            "irradiancia_w_m2": st.column_config.NumberColumn("Irradiancia (W/m²)", format="%.2f"),
            "velocidad_viento_m_s": st.column_config.NumberColumn("Viento (m/s)", format="%.2f"),
            "precipitacion_mm": st.column_config.NumberColumn("Precipitación (mm)", format="%.2f"),
            "generacion_solar_kwh": st.column_config.NumberColumn("Solar (kWh)", format="%.2f"),
            "generacion_eolica_kwh": st.column_config.NumberColumn("Eólica (kWh)", format="%.2f"),
            "generacion_pch_kwh": st.column_config.NumberColumn("PCH (kWh)", format="%.2f"),
            "demanda_kwh": st.column_config.NumberColumn("Demanda (kWh)", format="%.2f"),
            "balance_kwh": st.column_config.NumberColumn("Balance (kWh)", format="%.2f"),
            "decision": "Decisión",
        },
    )
    st.download_button(
        "Descargar datos filtrados (.csv)",
        data=filtrados.to_csv(index=False).encode("utf-8-sig"),
        file_name="comunidad_energetica_eafit_2026.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption("Datos sintéticos para fines académicos · EAFIT 2026 · Ciencia de datos")
