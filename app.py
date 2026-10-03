import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predicción de Nota Final", layout="wide")

st.title("Predicción de Aprobación de Curso")
st.write("Esta aplicación procesa las variables de entrada y realiza predicciones de la nota final utilizando el modelo pre-entrenado.")

# Opción de navegación en barra lateral
opcion = st.sidebar.selectbox("Selecciona el modo de uso:", ["Predicción Individual", "Cargar Archivo Excel (Lote)"])

# Cargar artefactos utilizando las rutas originales
try:
    one_hot_transformer = joblib.load('one_hot_columns.joblib')
    scaler = joblib.load('min_max_scaler.joblib')
    model = joblib.load('bagging_optimizado.joblib')
except Exception as e:
    st.error(f"Error crítico al cargar artefactos: {e}")
    st.stop()

# Helper para procesar y realizar predicciones
def procesar_y_predecir(df_input):
    df_procesado = df_input.copy()
    
    # Asegurar que se eliminen ID y Año - Semestre si están presentes
    columnas_a_eliminar = ['ID', 'Año - Semestre']
    df_procesado = df_procesado.drop(columns=[col for col in columnas_a_eliminar if col in df_procesado.columns], errors='ignore')
    
    # Codificación One-Hot de la columna 'Felder'
    if 'Felder' in df_procesado.columns:
        if isinstance(one_hot_transformer, list):
            si_columnas_one_hot = [col for col in one_hot_transformer if 'Felder_' in col]
            for col_name in si_columnas_one_hot:
                valor_esperado = col_name.replace('Felder_', '')
                df_procesado[col_name] = (df_procesado['Felder'].astype(str).str.strip() == valor_esperado).astype(float)
        else:
            df_encoded = pd.get_dummies(df_procesado[['Felder']])
            df_procesado = pd.concat([df_procesado, df_encoded], axis=1)
            si_columnas_one_hot = [col for col in df_procesado.columns if 'Felder_' in col]
            
        # Eliminar original
        df_procesado = df_procesado.drop(columns=['Felder'], errors='ignore')
    else:
        st.error("Falta la columna 'Felder' en los datos.")
        return None
        
    # Rellenar columnas faltantes del One-Hot si fuese necesario
    if isinstance(one_hot_transformer, list):
        for col in si_columnas_one_hot:
            if col not in df_procesado.columns:
                df_procesado[col] = 0.0
                
    # Escalar examen de admisión
    if 'Examen_admisión' in df_procesado.columns:
        df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[['Examen_admisión']])
        df_procesado = df_procesado.drop(columns=['Examen_admisión'], errors='ignore')
    else:
        st.error("Falta la columna 'Examen_admisión' en los datos.")
        return None
        
    # Reordenar columnas para que coincida exactamente con lo esperado
    columnas_finales = [col for col in one_hot_transformer if col in df_procesado.columns]
    df_procesado = df_procesado[columnas_finales]
    
    # Realizar predicción
    predicciones = model.predict(df_procesado)
    return df_procesado, predicciones

if opcion == "Predicción Individual":
    st.header("Predicción Individual (Manual)")
    
    opciones_felder = ['sensorial', 'activo', 'visual', 'equilibrio', 'secuencial', 'reflexivo', 'verbal', 'intuitivo']
    felder_input = st.selectbox("Selecciona el estilo de aprendizaje (Felder):", opciones_felder)
    examen_input = st.number_input("Examen de Admisión:", min_value=0.0, max_value=5.0, value=3.83, step=0.01)
    
    if st.button("Realizar Predicción Individual"):
        df_individual = pd.DataFrame([{'Felder': felder_input, 'Examen_admisión': examen_input}])
        result = procesar_y_predecir(df_individual)
        if result is not None:
            df_proc, pred = result
            st.subheader("Datos Procesados para el Modelo")
            st.dataframe(df_proc)
            st.success(f
