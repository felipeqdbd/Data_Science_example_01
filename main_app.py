"""Dashboard interactivo de generacion solar para una planta del Tolima."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Planta Solar Tolima",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def generar_datos(seed: int = 42, n_registros: int = 1_000) -> pd.DataFrame:
    """Genera mediciones horarias sinteticas con patrones solares realistas."""
    rng = np.random.default_rng(seed)
    fecha_hora = pd.date_range("2025-01-01", periods=n_registros, freq="h")
    hora = fecha_hora.hour.to_numpy()

    # Curva diurna: cero durante la noche y pico alrededor del mediodia.
    perfil_solar = np.maximum(0, np.sin(np.pi * (hora - 6) / 12))
    nubosidad = np.clip(rng.beta(5, 2, n_registros), 0.15, 1)
    irradiancia = np.clip(1_050 * perfil_solar * nubosidad + rng.normal(0, 22, n_registros), 0, None)
    temperatura = 22 + 10 * perfil_solar + rng.normal(0, 2.2, n_registros)

    # Planta de 5 MW. La temperatura reduce ligeramente el rendimiento.
    factor_temperatura = np.clip(1 - 0.004 * (temperatura - 25), 0.88, 1.06)
    potencia = np.clip(5_000 * (irradiancia / 1_000) * factor_temperatura + rng.normal(0, 55, n_registros), 0, 5_000)
    eficiencia = np.clip(19.2 * factor_temperatura + rng.normal(0, 0.45, n_registros), 15.5, 21.0)
    energia = np.clip(potencia * rng.uniform(0.94, 1.0, n_registros), 0, None)

    alarma = rng.random(n_registros) < np.where(potencia > 4_600, 0.09, 0.025)
    estado = np.full(n_registros, "Operando", dtype=object)
    estado[(irradiancia > 80) & (potencia < 250)] = "Rendimiento bajo"
    estado[alarma] = "Alerta"
    estado[irradiancia < 20] = "Sin generacion"

    return pd.DataFrame(
        {
            "fecha_hora": fecha_hora,
            "irradiancia_w_m2": irradiancia.round(2),
            "temperatura_c": temperatura.round(2),
            "potencia_kw": potencia.round(2),
            "energia_kwh": energia.round(2),
            "eficiencia_pct": eficiencia.round(2),
            "estado": pd.Categorical(
                estado,
                categories=["Operando", "Rendimiento bajo", "Alerta", "Sin generacion"],
            ),
            "alarma": alarma,
        }
    )


def formato_numero(valor: float, decimales: int = 1) -> str:
    return f"{valor:,.{decimales}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def descargar_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


st.title("☀️ Monitoreo de generación solar — Tolima")
st.caption(
    "Planta sintética de 5 MW · 1.000 mediciones horarias · datos generados dentro de Streamlit"
)

with st.sidebar:
    st.header("Controles")
    semilla = st.number_input("Semilla de simulación", min_value=0, max_value=10_000, value=42)
    if st.button("Regenerar datos", use_container_width=True, type="primary"):
        st.cache_data.clear()

datos = generar_datos(int(semilla))

with st.sidebar:
    st.subheader("Filtros")
    fecha_min = datos["fecha_hora"].min().date()
    fecha_max = datos["fecha_hora"].max().date()
    rango_fecha = st.date_input(
        "Rango de fechas",
        value=(fecha_min, fecha_max),
        min_value=fecha_min,
        max_value=fecha_max,
    )
    estados_disponibles = datos["estado"].dropna().astype(str).unique().tolist()
    estados = st.multiselect("Estado operativo", estados_disponibles, default=estados_disponibles)
    solo_alarmas = st.toggle("Mostrar solo alarmas", value=False)

    st.subheader("Umbrales")
    umbral_potencia = st.slider("Potencia mínima (kW)", 0, 5_000, 0, step=100)
    umbral_irradiancia = st.slider("Irradiancia mínima (W/m²)", 0, 1_100, 0, step=25)

    st.subheader("Personalización")
    paleta = st.selectbox("Paleta de color", ["Viridis", "Plasma", "Cividis", "Turbo", "Blues"])
    tema_grafico = st.selectbox("Tema de gráficos", ["plotly_white", "plotly_dark", "ggplot2", "seaborn"])

fecha_inicio, fecha_fin = fecha_min, fecha_max
if isinstance(rango_fecha, (tuple, list)) and len(rango_fecha) == 2:
    fecha_inicio, fecha_fin = rango_fecha
elif isinstance(rango_fecha, date):
    fecha_inicio = fecha_fin = rango_fecha

mascara = (
    datos["fecha_hora"].dt.date.between(fecha_inicio, fecha_fin)
    & datos["estado"].astype(str).isin(estados)
    & (datos["potencia_kw"] >= umbral_potencia)
    & (datos["irradiancia_w_m2"] >= umbral_irradiancia)
)
if solo_alarmas:
    mascara &= datos["alarma"]
filtrados = datos.loc[mascara].copy()

if filtrados.empty:
    st.warning("No hay registros para la combinación de filtros elegida. Ajusta los controles laterales.")
    st.stop()

energia_total = filtrados["energia_kwh"].sum()
potencia_media = filtrados["potencia_kw"].mean()
eficiencia_media = filtrados["eficiencia_pct"].mean()
alarmas_total = int(filtrados["alarma"].sum())

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Registros", f"{len(filtrados):,}".replace(",", "."), delta=f"de {len(datos):,}".replace(",", "."))
m2.metric("Energía total", f"{formato_numero(energia_total / 1_000)} MWh")
m3.metric("Potencia promedio", f"{formato_numero(potencia_media)} kW")
m4.metric("Eficiencia promedio", f"{formato_numero(eficiencia_media, 2)} %")
m5.metric("Alarmas", alarmas_total, delta=f"{alarmas_total / len(filtrados):.1%}", delta_color="inverse")

tab_resumen, tab_graficos, tab_estadistica, tab_datos = st.tabs(
    ["Resumen operativo", "Análisis gráfico", "Estadística", "Datos"]
)

numericas = [
    "irradiancia_w_m2",
    "temperatura_c",
    "potencia_kw",
    "energia_kwh",
    "eficiencia_pct",
]

with tab_resumen:
    izquierda, derecha = st.columns([2, 1])
    with izquierda:
        serie = filtrados.set_index("fecha_hora")["potencia_kw"]
        fig_serie = px.area(
            serie,
            labels={"fecha_hora": "Fecha y hora", "value": "Potencia (kW)", "variable": "Variable"},
            title="Potencia generada en el tiempo",
            color_discrete_sequence=["#F4A261"],
        )
        fig_serie.add_hline(
            y=umbral_potencia,
            line_dash="dash",
            line_color="#D62828",
            annotation_text="Umbral",
        )
        fig_serie.update_layout(template=tema_grafico, hovermode="x unified", showlegend=False)
        st.plotly_chart(fig_serie, use_container_width=True)
    with derecha:
        conteo_estado = filtrados["estado"].astype(str).value_counts().rename_axis("estado").reset_index(name="registros")
        fig_estado = px.pie(
            conteo_estado,
            names="estado",
            values="registros",
            hole=0.55,
            title="Distribución por estado",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_estado.update_layout(template=tema_grafico, legend_title_text="Estado")
        st.plotly_chart(fig_estado, use_container_width=True)

with tab_graficos:
    st.subheader("Explorador de variables")
    c1, c2, c3 = st.columns(3)
    x_var = c1.selectbox("Eje X", ["fecha_hora", *numericas], index=0)
    y_var = c2.selectbox("Eje Y", numericas, index=2)
    tipo = c3.selectbox("Tipo de gráfico", ["Línea", "Dispersión", "Barras", "Histograma", "Caja"])

    color_estado = st.checkbox("Colorear por estado operativo", value=tipo in {"Dispersión", "Barras"})
    color = "estado" if color_estado else None

    if tipo == "Línea":
        fig = px.line(filtrados.sort_values(x_var), x=x_var, y=y_var, color=color, markers=False)
    elif tipo == "Dispersión":
        tamano = st.selectbox("Variable para el tamaño de los puntos", ["Ninguna", *numericas], index=0)
        fig = px.scatter(
            filtrados,
            x=x_var,
            y=y_var,
            color=color,
            size=None if tamano == "Ninguna" else tamano,
            hover_data=["fecha_hora", "estado", "alarma"],
            opacity=0.72,
        )
    elif tipo == "Barras":
        agregacion = st.selectbox("Agregación", ["Promedio", "Suma", "Máximo", "Mínimo"])
        frecuencia = st.selectbox("Agrupar por", ["Día", "Semana", "Estado"])
        if frecuencia == "Estado":
            agrupador = filtrados["estado"].astype(str)
            etiqueta_x = "Estado"
        else:
            regla = "D" if frecuencia == "Día" else "W"
            agrupador = filtrados["fecha_hora"].dt.to_period(regla).dt.start_time
            etiqueta_x = frecuencia
        funcion = {"Promedio": "mean", "Suma": "sum", "Máximo": "max", "Mínimo": "min"}[agregacion]
        barras = filtrados.assign(grupo=agrupador).groupby("grupo", observed=True)[y_var].agg(funcion).reset_index()
        fig = px.bar(barras, x="grupo", y=y_var, color=y_var, color_continuous_scale=paleta)
        fig.update_xaxes(title=etiqueta_x)
    elif tipo == "Histograma":
        bins = st.slider("Número de intervalos", 10, 80, 30)
        fig = px.histogram(filtrados, x=y_var, color=color, nbins=bins, marginal="box")
    else:
        fig = px.box(filtrados, x="estado", y=y_var, color="estado", points="outliers")

    fig.update_layout(template=tema_grafico, height=540, legend_title_text="Estado")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Relaciones entre variables")
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
    fig_corr.update_layout(title="Matriz de correlación", template=tema_grafico, height=480)
    st.plotly_chart(fig_corr, use_container_width=True)

with tab_estadistica:
    st.subheader("Estadística cuantitativa")
    variable_estadistica = st.selectbox("Variable a analizar", numericas, index=2, key="estadistica")
    serie = filtrados[variable_estadistica]
    resumen = pd.DataFrame(
        {
            "Métrica": ["Conteo", "Media", "Mediana", "Desviación estándar", "Mínimo", "Q1", "Q3", "Máximo", "Coef. variación"],
            "Valor": [
                serie.count(), serie.mean(), serie.median(), serie.std(), serie.min(),
                serie.quantile(0.25), serie.quantile(0.75), serie.max(),
                serie.std() / serie.mean() if serie.mean() else np.nan,
            ],
        }
    )
    a, b = st.columns([1, 2])
    a.dataframe(resumen.style.format({"Valor": "{:,.2f}"}), use_container_width=True, hide_index=True)
    fig_dist = px.histogram(
        filtrados,
        x=variable_estadistica,
        color="estado",
        marginal="violin",
        nbins=35,
        title=f"Distribución de {variable_estadistica}",
    )
    fig_dist.update_layout(template=tema_grafico)
    b.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("Estadística cualitativa")
    cualitativo = (
        filtrados["estado"].astype(str).value_counts(dropna=False).rename("Frecuencia").to_frame()
    )
    cualitativo["Porcentaje"] = cualitativo["Frecuencia"] / len(filtrados) * 100
    cualitativo.loc["Total"] = [len(filtrados), 100.0]
    st.dataframe(cualitativo.style.format({"Porcentaje": "{:.2f}%"}), use_container_width=True)

with tab_datos:
    st.subheader("Dataset sintético filtrado")
    st.caption("Tipos: fecha/hora, cinco variables numéricas, una categoría y un booleano.")
    st.dataframe(
        filtrados,
        use_container_width=True,
        hide_index=True,
        column_config={
            "fecha_hora": st.column_config.DatetimeColumn("Fecha y hora", format="DD/MM/YYYY HH:mm"),
            "irradiancia_w_m2": st.column_config.NumberColumn("Irradiancia (W/m²)", format="%.2f"),
            "temperatura_c": st.column_config.NumberColumn("Temperatura (°C)", format="%.2f"),
            "potencia_kw": st.column_config.ProgressColumn("Potencia (kW)", min_value=0, max_value=5_000, format="%.0f"),
            "energia_kwh": st.column_config.NumberColumn("Energía (kWh)", format="%.2f"),
            "eficiencia_pct": st.column_config.NumberColumn("Eficiencia (%)", format="%.2f"),
            "estado": "Estado",
            "alarma": st.column_config.CheckboxColumn("Alarma"),
        },
    )
    st.download_button(
        "Descargar datos filtrados (.csv)",
        data=descargar_csv(filtrados),
        file_name="generacion_solar_tolima.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption("Los datos son completamente sintéticos y se generan con fines educativos.")

