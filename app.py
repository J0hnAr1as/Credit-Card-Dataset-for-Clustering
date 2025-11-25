import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Credit Card Clustering", layout="wide")

# --- CARGA DE MODELOS ---
@st.cache_resource
def load_data():
    try:
        model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        pca = joblib.load('pca_model.pkl')
        pca_data = pd.read_csv('pca_labeled_data.csv', dtype={'Cluster': str})
        return model, scaler, pca, pca_data
    except:
        return None, None, None, None

kmeans, scaler, pca_model, pca_base_data = load_data()

# --- DESCRIPCIONES ---
cluster_desc = {
    '0': "Bajo Consumo: Límites bajos, poco uso.",
    '1': "VIP/Alto Gasto: Compras altas y frecuentes.",
    '2': "Deudores de Efectivo: Muchos avances, pago mínimo.",
    '3': "Uso Promedio: Comportamiento estándar."
}

# --- UI ---
st.title("💳 Segmentación de Clientes (K-Means + PCA)")

if kmeans is None:
    st.error("⚠️ Faltan los archivos del modelo. Ejecuta 'python train_model.py' primero.")
else:
    with st.sidebar:
        st.header("Datos del Cliente")
        # Inputs simplificados para la demo
        balance = st.number_input("Balance Actual", 0.0, value=1000.0)
        purchases = st.number_input("Monto Compras ($)", 0.0, value=500.0)
        cash_adv = st.number_input("Avances Efectivo ($)", 0.0, value=0.0)
        cred_lim = st.number_input("Límite Crédito", 0.0, value=5000.0)
        payments = st.number_input("Pagos Realizados", 0.0, value=1000.0)
        
        # Inputs avanzados colapsables
        with st.expander("Ver configuración avanzada (12 variables más)"):
            bal_freq = st.slider("Frecuencia Balance", 0.0, 1.0, 0.8)
            oneoff = st.number_input("Compras One-Off", 0.0, value=0.0)
            install = st.number_input("Compras Cuotas", 0.0, value=200.0)
            pur_freq = st.slider("Frecuencia Compras", 0.0, 1.0, 0.5)
            one_freq = st.slider("Frecuencia One-Off", 0.0, 1.0, 0.0)
            inst_freq = st.slider("Frecuencia Cuotas", 0.0, 1.0, 0.5)
            cash_freq = st.slider("Frecuencia Avances", 0.0, 1.0, 0.0)
            cash_trx = st.number_input("Transacciones Avances", 0, value=0)
            pur_trx = st.number_input("Transacciones Compras", 0, value=10)
            min_pay = st.number_input("Pago Mínimo", 0.0, value=200.0)
            prc_full = st.slider("Porcentaje Pago Total", 0.0, 1.0, 0.0)
            tenure = st.selectbox("Tenure", [6, 12], index=1)

        btn_calc = st.button("Calcular Cluster", type="primary")

    if btn_calc:
        # Crear DataFrame (orden estricto)
        row = [balance, bal_freq, purchases, oneoff, install, cash_adv, pur_freq, 
               one_freq, inst_freq, cash_freq, cash_trx, pur_trx, cred_lim, 
               payments, min_pay, prc_full, tenure]
        
        cols = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
                'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
                'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
                'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
                'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']
        
        df_in = pd.DataFrame([row], columns=cols)
        
        # Predecir
        X_scaled = scaler.transform(df_in)
        cluster = str(kmeans.predict(X_scaled)[0])
        
        # Visualizar
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.metric(label="Cluster Asignado", value=cluster)
            st.info(cluster_desc.get(cluster))
            
        with col2:
            # Proyectar punto nuevo
            pca_pt = pca_model.transform(X_scaled)
            
            fig = px.scatter(pca_base_data, x='PC1', y='PC2', color='Cluster', 
                             title="Mapa de Clusters", opacity=0.3, 
                             color_discrete_sequence=px.colors.qualitative.Set1)
            
            fig.add_trace(go.Scatter(x=[pca_pt[0,0]], y=[pca_pt[0,1]], 
                                     mode='markers', name='Nuevo Cliente',
                                     marker=dict(color='black', size=20, symbol='x')))
            
            st.plotly_chart(fig, use_container_width=True)