import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Predicción de Aprobación de Curso")
st.write("Esta aplicación procesa las variables de entrada y realiza una predicción utilizando un modelo de Bagging pre-entrenado.")

# Cargar los recursos (modelos y transformadores) una sola vez
@st.cache_resource
def load_resources():
    one_hot_transformer = joblib.load('one_hot_columns.joblib')
    scaler = joblib.load('min_max_scaler.joblib')
    model = joblib.load('bagging_optimizado.joblib')
    return one_hot_transformer, scaler, model

try:
    one_hot_transformer, scaler, model = load_resources()
except Exception as e:
    st.error(f"Error al cargar los modelos pre-entrenados: {e}")
    st.stop()

# Función modular para procesar datos (sirve para manual y excel)
def procesar_datos(df_input):
    df_procesado = df_input.copy()
    
    # 1. Eliminar columnas innecesarias si existen
    columnas_a_eliminar = ['ID', 'Año - Semestre', 'Nota_final', 'Aprobo']
    df_procesado = df_procesado.drop(columns=[col for col in columnas_a_eliminar if col in df_procesado.columns], errors='ignore')
    
    # 2. One-Hot Encoding para 'Felder'
    si_columnas_one_hot = [col for col in one_hot_transformer if 'Felder_' in col] if isinstance(one_hot_transformer, list) else []
    if 'Felder' in df_procesado.columns:
        for col_name in si_columnas_one_hot:
            valor_esperado = col_name.replace('Felder_', '')
            df_procesado[col_name] = (df_procesado['Felder'] == valor_esperado).astype(int)
        df_procesado = df_procesado.drop(columns=['Felder'])
    
    # Asegurar que existan todas las columnas One-Hot (por si faltan)
    for col in si_columnas_one_hot:
        if col not in df_procesado.columns:
            df_procesado[col] = 0
            
    # 3. Normalizar 'Examen_admisión'
    if 'Examen_admisión' in df_procesado.columns:
        df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[['Examen_admisión']])
        df_procesado = df_procesado.drop(columns=['Examen_admisión'])
        
    # 4. Reordenar las columnas al formato esperado por el modelo
    columnas_ordenadas = si_columnas_one_hot + ['Examen_admision_scaled']
    
    # Rellenar con 0 cualquier columna faltante (seguridad extra)
    for col in columnas_ordenadas:
        if col not in df_procesado.columns:
            df_procesado[col] = 0
            
    return df_procesado[columnas_ordenadas]

# Crear pestañas para la UI
tab1, tab2 = st.tabs(["Ingreso Manual", "Subir Archivo Excel"])

with tab1:
    st.header("Datos de Entrada Manual")
    opciones_felder = ['sensorial', 'activo', 'visual', 'equilibrio', 'secuencial', 'reflexivo', 'verbal', 'intuitivo']
    felder_input = st.selectbox("Selecciona el estilo de aprendizaje (Felder):", opciones_felder)
    examen_input = st.number_input("Examen de Admisión:", min_value=0.0, max_value=5.0, value=3.83, step=0.01)

    if st.button("Realizar Predicción Individual"):
        df_input_manual = pd.DataFrame({'Felder': [felder_input], 'Examen_admisión': [examen_input]})
        try:
            df_procesado = procesar_datos(df_input_manual)
            st.subheader("Datos Procesados para el Modelo")
            st.dataframe(df_procesado)
            
            prediccion = model.predict(df_procesado)
            st.success(f"La predicción del modelo (Nota Final Estimada) es: {prediccion[0]:.4f}")
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")

with tab2:
    st.header("Predicción por Lotes (Archivo Excel)")
    archivo_subido = st.file_uploader("Sube un archivo Excel (.xlsx) con los datos de prueba:", type=["xlsx", "xls"])
    
    if archivo_subido is not None:
        try:
            df_excel = pd.read_excel(archivo_subido)
            st.write("Vista previa de los datos originales:")
            st.dataframe(df_excel.head())
            
            if st.button("Realizar Predicción por Lotes"):
                with st.spinner('Procesando y prediciendo...'):
                    # Procesar los datos completos
                    df_procesado_batch = procesar_datos(df_excel)
                    
                    # Realizar predicciones
                    predicciones = model.predict(df_procesado_batch)
                    
                    # Crear dataframe de resultados
                    df_resultados = df_excel.copy()
                    df_resultados['Predicción_Nota_Final'] = predicciones
                    
                    st.success("¡Predicciones completadas con éxito!")
                    st.write("Resultados con las predicciones añadidas:")
                    st.dataframe(df_resultados)
                    
        except Exception as e:
            st.error(f"Error al procesar el archivo Excel: {e}")
