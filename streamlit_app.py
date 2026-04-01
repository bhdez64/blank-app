import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from io import BytesIO

st.set_page_config(page_title="Feeder Analysis", layout="wide")

st.title("Feeder Reliability Analysis Dashboard")

uploaded_file = st.file_uploader("Upload a file Excel o CSV", type=["xlsx", "csv"])

def load_file(file):
    if file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    return pd.read_csv(file)

def export_excel(dataframe):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False)
    return output.getvalue()

if uploaded_file is not None:

    df = load_file(uploaded_file)
    df.columns = [str(col).strip() for col in df.columns]

    st.subheader("Preview of data")
    st.dataframe(df.head())

    # Validación
    required = ["Substation", "Feeder", "Outage #", "SAIDI", "Customers Out"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        st.error(f"Faltan columnas: {missing}")
        st.stop()

    # Convertir a numérico
    for col in ["Substation", "Feeder", "SAIDI", "Customers Out"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Substation", "Feeder"])

    # Filtro por causa
    if "Cause" in df.columns:
        causas = ["Todas"] + list(df["Cause"].dropna().unique())
        seleccion = st.sidebar.selectbox("Filter by cause", causas)

        if seleccion != "Todas":
            df = df[df["Cause"] == seleccion]

    # Agrupación
    resumen = df.groupby(["Substation", "Feeder"]).agg({
        "Outage #": "count",
        "SAIDI": "sum",
        "Customers Out": "sum"
    }).reset_index()

    resumen = resumen.rename(columns={
        "Outage #": "faults",
        "Customers Out": "affected_customers"
    })

    resumen["feeder"] = (
        resumen["Substation"].astype(int).astype(str) +
        "-" +
        resumen["Feeder"].astype(int).astype(str)
    )

    # KPI
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Feeders", len(resumen))
    col2.metric("Faults", int(resumen["faults"].sum()))
    col3.metric("SAIDI total", round(resumen["SAIDI"].sum(), 2))
    col4.metric("Affected Customers", int(resumen["affected_customers"].sum()))

    # Clustering
    X = resumen[["SAIDI", "faults", "affected_customers"]].fillna(0)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    resumen["group"] = kmeans.fit_predict(X)

    # Riesgo
    resumen["risk"] = (
        resumen["SAIDI"] * 0.5 +
        resumen["faults"] * 0.3 +
        resumen["affected_customers"] * 0.2
    )

    # Layout
    col_g1, col_g2 = st.columns(2)

    # Gráfico barras
    with col_g1:
        st.subheader("Top 10 feeders (SAIDI)")
        top = resumen.sort_values(by="SAIDI", ascending=False).head(10)

        fig1, ax1 = plt.subplots()
        ax1.bar(top["feeder"], top["SAIDI"])
        plt.xticks(rotation=45)
        st.pyplot(fig1)

    # Scatter clustering
    with col_g2:
        st.subheader("Clustering of feeders")

        fig2, ax2 = plt.subplots()
        scatter = ax2.scatter(resumen["SAIDI"], resumen["faults"], c=resumen["group"])

        for i in range(len(resumen)):
            ax2.text(
                resumen["SAIDI"].iloc[i],
                resumen["faults"].iloc[i],
                resumen["feeder"].iloc[i],
                fontsize=7
            )

        ax2.set_xlabel("SAIDI")
        ax2.set_ylabel("Faults")

        st.pyplot(fig2)

    # Ranking
    st.subheader("Feeders more critical")
    top_riesgo = resumen.sort_values(by="risk", ascending=False).head(10)
    st.dataframe(top_riesgo)

    # Modelo predictivo
    st.subheader("Pedictive Model")

    resumen["critical"] = ((resumen["faults"] > 10) | (resumen["SAIDI"] > 100)).astype(int)

    X = resumen[["SAIDI", "faults", "affected_customers"]]
    y = resumen["critical"]

    if len(resumen) > 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)

        st.write("Accuracy:", round(acc, 2))

        resumen["prediction"] = model.predict(X)

        st.dataframe(resumen[[
            "feeder", "faults", "SAIDI",
            "affected_customers", "prediction"
        ]])

    # Exportar
    excel = export_excel(resumen)

    st.download_button(
        "Download Excel",
        excel,
        file_name="Feeders_analysis.xlsx"
    )

else:
    st.info("Upload a file to start")