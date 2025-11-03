# ========================================================================
# SISTEMA MULTI-MODAL PARA CHLORELLA VULG EN FOTOBIORREACTORES
# ========================================================================
# Sistema para predicción de biomasa de Chlorella vulgaris
# con detección automática de data leakage, validación temporal y
# análisis estadístico
#
# 1. GESTIÓN INTELIGENTE DE DATOS (SmartDataManager)
#    - Detección automática de data leakage
#    - Umbral de correlación (0.95)
#    - Validación de variables biológicas
#    - Limpieza y preprocesamiento
#
# 2. INGENIERÍA DE CARACTERÍSTICAS BIOLÓGICAS (BioFeatureEngine)
#    a) Variables Fotosintéticas
#       - Eficiencia lumínica (Michaelis-Menten)
#       - Fotoinhibición
#       - Penetración de luz
#    b) Variables Ambientales
#       - Efectos de temperatura y pH (funciones gaussianas)
#       - Estrés ambiental
#       - Interacciones multi-factor
#    c) Dinámica de Nutrientes
#       - Limitación por nutrientes (función sigmoide)
#       - Efecto Monod
#    d) Variables Temporales
#       - Ciclos diurnos
#       - Fases de crecimiento
#
# 3. SELECCIÓN DE CARACTERÍSTICAS (3 métodos)
#    - Correlación de Pearson
#    - SelectKBest (f_regression)
#    - Random Forest (50 árboles)
#    → Ensemble de los tres métodos
# 
# 4. DIVISIÓN DE DATOS , REPRODUCIBILIDAD Y NORMALIZACIÓN
#
#   - División por escenarios (75% entrenamiento, 25% validación)
#   - Semilla aleatoria fija (50)
#   - Normalización robusta (RobustScaler)
#   - Estandarización (StandardScaler)
#   
#
# 5. SISTEMA MULTI-MODELO
#    a) Modelos Base
#       - Regresión Lineal
#       - Ridge
#       - Random Forest
#    b) Modelos Avanzados
#       - PINN (Physics-Informed Neural Network)
#       - LSTM (Long Short-Term Memory)
#       - XGBoost
#    c) Ensemble Ponderado
#       - Pesos basados en rendimiento
#       - Validación temporal
#
# 6. EVALUACIÓN Y VALIDACIÓN
#    - Métricas: R², RMSE, MAE, MAPE, NSE, Bias
#    - Análisis de residuos
#    - Detección de overfitting
#    - Visualizaciones estadísticas
#
# CARACTERÍSTICAS ANTI-OVERFITTING:
# - Validación temporal (en la división por escenarios 45/15)
# - Detección de data leakage
# - Ensemble de modelos
# - Regularización L2 (Ridge Regression y AdamW)


# =======================================================================
# inicio del código:
# ========================================================================
# importo las librerías necesarias:

# pandas: para manipulación de datos
# numpy: para operaciones numéricas
# torch: para modelos de aprendizaje profundo
# sklearn: para preprocesamiento y métricas
# xgboost: para modelos de boosting
# matplotlib y seaborn: para visualización
# datetime: para manejo de fechas
# warnings: para manejar advertencias
# scipy: para estadísticas y pruebas
# scipy.stats: para pruebas estadísticas avanzadas
# statsmodels: para herramientas estadísticas avanzadas

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("CHLORELLA VULGARIS SISTEMA MULTI-MODAL")
# MUESTRO fecha y hora de inicio
#print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ========================================================================
# PASO 1: CARGA Y PROCESAMIENTO INTELIGENTE
# ========================================================================

#----------------------------------------------------------------------
# Concepto: data leakage (fuga de datos)
#----------------------------------------------------------------------
# El data leakage ocurre cuando el modelo tiene acceso a información
# que no debería tener durante el entrenamiento, lo que puede llevar
# a una sobreestimación de su rendimiento. Por ejemplo, si el modelo
# tiene acceso a valores futuros que no estarían disponibles en el
# momento de la predicción.

# Para evitar esto necesito:
# 1. Separar los datos de entrenamiento y prueba antes de cualquier preprocesamiento.
# 2. No incluir variables que contengan información futura o que sean
#    directamente derivadas del objetivo a predecir.
# 3. Validar que las variables de entrada no contengan información que
#    pueda filtrar el objetivo de manera directa.
#----------------------------------------------------------------------


# leakage_threshold es un umbral de correlación que se usa para detectar
# data leakage o fuga de datos 

# la correlacion nos indica la fuerza de la relación lineal entre dos variables
# si es cercana a 1, significa que una variable está directamente relacionada
# con la variable objetivo, lo que puede indicar fuga de datos.

# es decir, si una variable tiene una correlación alta con la variable objetivo,
# es probable que esté filtrando información que no debería estar disponible, 
# la variable podría estar utilizando información futura o derivada del objetivo 
# para hacer predicciones y eso es lo que queremos evitar.

class SmartDataManager:
    def __init__(self, leakage_threshold=0.95):

        # Inicializo el umbral de correlación para detectar data leakage
        # normalmente se usa un valor alto, de forma arbitraria, de 0.95 y se usa porque,
        # como ya expliqué, correlaciones cercanas a 1 indican que la variable esta 
        # directamente relacionada con la variable objetivo (cosa a evitar)

        self.leakage_threshold = leakage_threshold

        # guardo el valor por defecto de leakage_threshold 0.95
        
    def carga_limpia_datos(self, filename='complete_dataset.csv'):
       # Carga, valida y procesa datos en una función
        try:
            #creo un nuevo DataFrame vacío (data)
            # pd.read_csv carga el dataset complete_dataset.csv y lo almacena en el DataFrame llamado data
            data = pd.read_csv(filename)

            print(f"✅ Dataset: {data.shape}") #muestro el tamaño del dataset (filas, columnas)

            # ----------------------------------------------------
                # Detectar data leakage automáticamente
            # ----------------------------------------------------   

            leakage_cols = self._detecta_leakage(data)

            # leakage_cols son las columnas (variables) que pueden causar fuga de datos
            #-----------------
            # _detecta_leakage
            #-----------------
             # lo que hace esta funcion es analizar las columnas del dataframe y detectar las variables
             # que tienen una alta correlacion con la variable objetivo (biomasa)
             # se queda con las variables que superen el umbral de 0.95 establecido en el punto anterior
            
            if leakage_cols:
                print(f"🚫 Eliminando {len(leakage_cols)} características que producen fuga de datos")
                # En caso de detectar alguna variable que cause fuga de datos,
                # se imprime un mensaje indicando cuántas variables se eliminarán
                data = data.drop(columns=leakage_cols)
                # y se eliminan las columnas que causan fuga de datos del nuevo DataFrame

            return data, leakage_cols
        # devuelvo el dataframe limpio sin las columnas que causan fuga de datos y también 
        # devuelvo una lista de las columnas que causan fuga de datos

        # en caso de que no se encuentre el archivo
        except FileNotFoundError:
            print("❌ Dataset no encontrado. Asegúrate de que el archivo 'complete_dataset.csv' esté en el directorio correcto.")
            return None, []

    def _detecta_leakage(self, data):
       # funcion añadida para detectar data leakage
        # recibe un DataFrame y devuelve una lista de columnas que pueden causar fuga de datos


        # Verifico si la columna 'Biomass_g_L' está presente
        # ya que es la variable objetivo y si no está, no puedo detectar fuga de datos
        if 'Biomass_g_L' not in data.columns:
            return [] # devuelvo una lista vacía si no está la columna 'Biomass_g_L'
        # asi en vez de lanzar un error, simplemente devuelvo una lista vacía y el codigo sigue funcionando
        
        # asigno las variables que sé seguro (por conocimiento biologico) que podrían causar fuga de datos 
        # pues son las variables del dataFrame que se relacionan directamente con la biomasa

        conocidas = ['Cell_Concentration_cells_mL', 'Cell_Density_10E6_mL', 
                'Instantaneous_Productivity_g_L_d', 'Specific_Growth_Rate_h']
        # ahora realizo una seleccion inteligente de columnas numéricas del DataFrame
        cols_numericas = data.select_dtypes(include=[np.number]).columns.tolist()
        # data.select_dtypes() es un metodo de la libreria pandas que filtra las columnas de un DataFrame
        # en funcion del tipo de dato que contiene
        # incluye los enteros y decimales y excluye texto, fechas y booleanos
        # .columns devuelve los nombres de las columnas seleccionadas

        # finalmente me quedo con un indice con los nombres de todas las columnas numéricas
        # y calculo la correlación entre las columnas numéricas y la columna 'Biomass_g_L'
        

        if len(cols_numericas) > 1:
            # en caso de que haya mas de una columna numerica
            corr = data[cols_numericas].corrwith(data['Biomass_g_L']).abs()
            # .corrwith(biomasa) calcula la correlacion de cada variable con la biomasa
            # .abs() devuelve el valor absoluto de la correlación, para evitar valores negativos
            # se guardan los datos de las correlaciones en una lista llamada corr

            corr_alta = corr[corr > self.leakage_threshold].index.tolist()
            # para cada variable, en caso de que su correlación con la biomasa sea mayor al umbral de 0.95
            # se guarda el nombre de la variable en una lista llamada corr_alta

            if 'Biomass_g_L' in corr_alta:
                corr_alta.remove('Biomass_g_L')
                # la biomasa siempre tiene correlacion 1 con si misma, 
                # con corr_alta.remove('Biomass_g_L') elimino la biomasa de mi ultima lista pues no es una variable que me influya al ser la variable objetivo
        else:
            corr_alta = []
            # si por algun casual ya no estaba en la lista devuelvo una lista vacia (error elegante en progr.)

        suspected = list(set(conocidas + corr_alta))
        # lo siguiente es juntar las variabes biologicas que ya sé que pueden causar DL (conocidas) y las variables que superan el umbral de correlación (corr_alta)
        # set() elimina duplicados que puedan aparecer en ambas listas y luego convierto a una nueva lista

        problematicas = [col for col in suspected if col in data.columns]

        # recorro columna a columna la lista de variables sospechosas
        # y compruebo que estén en el dataset original, luego las guardo en una nueva lista (problematicas)
        if problematicas:
            print(f"⚠️ Leakage detectado: {problematicas}")
        
        return problematicas
    
   
# ========================================================================
# PASO 2: FEATURE ENGINEERING BIOLÓGICO
# ========================================================================

# El objetivo es crear un conjunto de datos con muchas variables que mejoren
# la capacidad predictiva del modelo, para que pueda aprender patrones complejos

#crear_features:
# Partiendo del Dataset (tabla con columnas temperatura, pH, luz, biomasa, etc)
# la idea es añadir nuevas columnas al dataset que representan características útiles para predecir la biomasa
# (Biomass_g_L). Es como añadir nuevas formas de describir los datos para que el modelo
# entienda mejor cómo se relacionan con el objetivo 
class BioFeatureEngine:
    def __init__(self): #inicializo la clase BioFeatureEngine

# Un outlier es un valor atípico que puede aparecer en el conjunto de datos que se desvía
# significativamente de otros valores en el conjunto de datos. 
# Los outliers pueden distorsionar los resultados del modelo
#--------------------------------------------------------------------
# RobustScaler para manejar outliers y normalizar
#-------------------------------------------------------------------
# RobustScaler es una técnica de escalado que utiliza la mediana y el rango intercuartílico
# para escalar las características, lo que lo hace robusto a los outliers
# es decir, transforma los datos para que tengan una media de 0 y una desviación estándar de 1
# Esto es útil para que los modelos de aprendizaje automático funcionen mejor
        
# self.scalers es un diccionario que contiene dos escaladores:
# features: son las variables de entrada que se utilizan para predecir el objetivo
# target: es la variable que se quiere predecir, en este caso Biomass_g_L

        self.scalers = {'features': RobustScaler(), 'target': StandardScaler()}
        # Inicializo un atributo para almacenar las features seleccionadas
        self.selected_features = None

#incorporo una copia del DataFrame data para trabajar sobre el sin modificar el original      
    def create_features(self, data):
        df = data.copy()
        print("🧬 Creando features biológicas...")
        
        # Variables fotosintéticas O FEATURES añadidas para aportar informacion al modelo


        # ----------------------------------------------------
        # ECUACIÓN de Michaelis-Menten:
        # ----------------------------------------------------

        # La eficiencia de la luz se calcula como la tasa de fotosintesis
        # en función de la intensidad de luz (PAR) y una constante de saturacion
        
        # -------------------------
        # P = (I * Pmax) / (I + K)
        # -------------------------

        # P es la tasa de fotosíntesis, con un valor normalizado entre 0 y 1
        # I es la intensidad de luz 
        # K es la constante de semisaturación
        # Pmax es la tasa máxima de fotosíntesis, siendo el valor maximo normalizado 1 

        # K representa la intensidad de luz donde la tasa de fotosíntesis es la mitad de su valor máximo, 150 µmol/m²/s

        # Falkowski, P. G., & Raven, J. A. (2013). Aquatic Photosynthesis. Princeton University Press.

        # Curvas P-I para microalgas y fitoplancton, obtuvo valores de entre 50 y 200 µmol/m²/s para especies comunes como Chlorella.
        #  El valor de 150 es un promedio comun para microalgas en cultivos controlados.

        # 2. Jassby-Platt para eficiencia lumínica
        alpha_jp = 0.012  # Eficiencia inicial
        Pmax_jp = 1.0     # Tasa máxima normalizada
        df['eficiencia_luminica__jassby_platt'] = Pmax_jp * np.tanh((alpha_jp * df['PAR_umol_m2_s']) / Pmax_jp)

        #-----------------------------------------------------
        # ECUACION DE LA FOTOINHIBICIÓN:
        #-----------------------------------------------------

        # La fotoinhibición es el daño a la fotosíntesis causado por una luz excesiva
        # Se modela como una función lineal de la intensidad de luz que excede un umbral
        # En este caso, se considera que la fotoinhibición comienza a ocurrir por encima de 300 µmol/m²/s
        # y aumenta linealmente hasta 400 µmol/m²/s, donde se considera máxima
        
        # -------------------------------------
        # F = ( PAR - 300) / 100 si PAR > 300
        # -------------------------------------


        # dividido por 100 para obtener un valor normalizado entre 0 y 1 de la fotoinhibición
        
        # F = 0 si PAR <= 300
        # por debajo de 300 no se produce fotoinhibición

        # Referencias:
        # Long, S. P., Humphries, S., & Falkowski, P. G. (1994). fotoinhibicion of photosynthesis in nature. Annual Review of Plant Biology, 45(1), 633-662.
        # - Menciona que microalgas como Chlorella pueden mostrar fotoinhibición a partir de 250-300 µmol/m²/s en cultivos densos.

        # Tredici, M. R. (2010). Photobiology of microalgae mass cultures: understanding the tools for the next green revolution. Biofuels, 1(1), 143-162
        # - Explica cómo la fotoinhibición limita la productividad en cultivos de microalgas en fotobiorreactores, especialmente a intensidades de luz superiores a 200-400 µmol/m²/s.
        

        
        # np.maximum(0, x) compara x con 0 y devuelve el máximo entre ambos 

        df['fotoinhibicion'] = np.maximum(0, (df['PAR_umol_m2_s'] - 300) / 100)
        
        # ------------------------------------------------------
        # EFECTOS DE TEMPERATURA Y pH
        # ------------------------------------------------------

        # Las funciones gaussianas modelan cómo el crecimiento de microalgas (medido como Biomass_g_L) 
        # responde a la temperatura y el pH, con un pico en condiciones óptimas y una caída simétrica
        # en condiciones subóptimas
        # es decir, la función gaussiana me permite modelar cómo cambia la biomasa en función de la temperatura y el pH
        # La temperatura óptima es 28°C y el pH óptimo es 8.0

        # -----------------------------------------
        # función gaussiana:
        # temp_efecto = exp(-(temp - μ)² / (2σ²))
        # -----------------------------------------

        # donde μ es el valor óptimo (28°C o 8.0 pH) y σ controla la amplitud de la curva
        # La amplitud determina cuán rápido disminuye la biomasa a medida que nos alejamos de la temper


        # muchas microalgas toleran desviaciones de 5-10°C antes de un colapso significativo, 
        # en el caso de la temperatura he decidido darle una amplitud σ de 5, 
        # Esto significa que el crecimiento de microalgas es tolerante a desviaciones de ±5°C
        # desde el óptimo de 28°C (rango 23-33°C), con una reducción significativa del
        # crecimiento a ±10°C

        # Raven & Geider (1988) indican que el crecimiento disminuye simétricamente alrededor 
        # del óptimo (~28°C), con tolerancia a desviaciones de 5-10°C

        df['efecto_temp'] = np.exp(-((df['Temperature_C'] - 28)**2) / 50)

        # El pH óptimo para microalgas es 7.5-8.5, y desviaciones de ±0.5-1.0 unidades afectan 
        # la disponibilidad de carbono y el metabolismo. Una σ de 1 refleja esta alta sensibilidad

        # pH_efecto = exp(-(pH - μ)² / (2σ²))

        # Goldman & Azam (1978) muestran que desviaciones de ±0.5 unidades
        # desde pH 8.0 reducen significativamente la fotosíntesis.

        df['efecto_pH'] = np.exp(-((df['pH'] - 8.0)**2) / 2)

        
        # ----------------------------------------------------
        # ESTRES AMBIENTAL:
        # ----------------------------------------------------

        # El estrés ambiental se calcula como la desviación de la temperatura y pH óptimos
        # La temperatura óptima es 28°C y el pH óptimo es 8.0
        # Se mide como la distancia absoluta a estos valores óptimos, normalizada
        

        # La diferencia absoluta (|Temperature_C - 28|) mide cuánto se aleja la temperatura
        # del punto ideal para el crecimiento de microorganismos fotosintéticos

        df['estres_de_temperatura'] = np.abs(df['Temperature_C'] - 28)

        # La diferencia absoluta (|pH - 8.0|) mide cuánto se aleja el pH del punto ideal

        df['estres_ph'] = np.abs(df['pH'] - 8.0)

        # La suma ponderada de ambos da una medida compuesta de estrés ambiental
        # asumo que el estrés por pH (/10) tiene un peso relativo cinco veces mayor que el estrés por temperatura (/2) esto lo corroboraré por ahora lo dejaré así
        
        df['estres_ambiental'] = df['estres_de_temperatura']/10 + df['estres_ph']/2
        
         # referencias:
        # Eppley, R. W. (1972). Temperature and phytoplankton growth in the sea. Fishery Bulletin, 70(4), 1063-1085.
        # Relevancia: el crecimiento de las microalgas es menos sensible a la temperatura que a otros factores como el pH o la luz en rangos típicos de cultivo.
        # - Menciona que la temperatura óptima para microalgas como Chlorella es de 25-30°C, y el pH óptimo es de 7.5-8.5.

        # Raven, J. A., & Geider, R. J. (1988). Temperature and algal growth. New Phytologist, 110(4), 441-461.
        # -Indica que las microalgas toleran rangos de temperatura más amplios (20-35°C) que rangos de pH. Una desviación de 5°C del óptimo reduce el crecimiento, pero no tanto como una desviación de 0.5 unidades de pH.
        
        #Hinga, K. R. (2002). Effects of pH and temperature on phytoplankton physiology. Journal of Plankton Research, 24(12), 1201-1216.
        # Relevancia: Sugiere que la temperatura tiene un impacto más gradual que el pH, justificando un factor de normalización menor (como 1/10)
        
        
        # ----------------------------------------------------
        # ECUACIONES DE DINÁMICA DE NUTRIENTES:
        # ----------------------------------------------------

        if 'Nutrients_g_L' in df.columns:

            # 1. Limitación por nutrientes (Función sigmoide)
            # ----------------------------------------------
            # L = 1 / (1 + exp(5 * (N - 0.5)))
            # ----------------------------------------------
            # - N es la concentración de nutrientes (g/L)
            # - 0.5 es el punto medio de la curva
            # - 5 es el factor que controla la pendiente
            #
            # La función devuelve un valor entre 0 y 1:
            # - Cerca de 1 cuando hay pocos nutrientes (limitación alta)
            # - Cerca de 0 cuando hay muchos nutrientes (limitación baja)
            
            #df['nutrient_limitation'] = 1 / (1 + np.exp(5 * (df['Nutrients_g_L'] - 0.5)))
            
            # 2. Efecto de los nutrientes (Monod)
            # ----------------------------------
            # E = N / (N + Ks)
            # ----------------------------------
            # - N es la concentración de nutrientes
            # - Ks = 0.02 es la constante de semisaturación
            #
            # La función modela el efecto de los nutrientes en el crecimiento:
            # - Tiende a 1 con alta concentración de nutrientes
            # - Tiende a 0 cuando hay pocos nutrientes
            #
            # Referencias:
            # - Monod, J. (1949). The growth of bacterial cultures. Annual Review of Microbiology, 3(1), 371-394.
            # - Bernard, O. (2011). Hurdles and challenges for modelling and control of microalgae for CO2 mitigation 
            #   and biofuel production. Journal of Process Control, 21(10), 1378-1389.
            # 1. Haldane para nutrientes (incluye inhibición por exceso)
            Ks_haldane = 0.02  # Constante de semisaturación
            Ki_haldane = 1.5   # Constante de inhibición
            df['efecto_de_nutrientes_haldane'] = df['Nutrients_g_L'] / (Ks_haldane + df['Nutrients_g_L'] + (df['Nutrients_g_L']**2 / Ki_haldane))
            
        
        # ----------------------------------------------------
        # VARIABLES CICLICAS Y DE FASES DE CRECIMIENTO
        # ----------------------------------------------------
        # con esto creo un reloj de 6h para capturar ciclos de dia y noche con la función seno y coseno
        
        #El seno y el coseno convierten el tiempo en valores que "vuelven al inicio" cada 24 horas. 
        # Esto ayuda al modelo a entender que el tiempo es un ciclo continuo

        # 2π es una constante matemática (aproximadamente 6.28) que asegura que el ciclo se complete cada 24 horas. Dividir Time_h entre 24 normaliza el tiempo a un ciclo.
        # es importante usar ambos (seno y coseno) porque juntos capturan todas las posiciones del ciclo:
        # A las 0 horas (medianoche): sin_24h = 0, cos_24h = 1.
        # A las 6 horas: sin_24h = 1, cos_24h = 0.
        # A las 12 horas: sin_24h = 0, cos_24h = -1.
        # A las 18 horas: sin_24h = -1, cos_24h = 0.

        df['sin_24h'] = np.sin(2 * np.pi * df['Time_h'] / 24)
        df['cos_24h'] = np.cos(2 * np.pi * df['Time_h'] / 24)
        
        # Sin estas transformaciones, el modelo podría malinterpretar el tiempo 
        # (por ejemplo, pensar que las 23:00 del dia 1 están muy lejos de las 00:00 del día 2)
        # es para que el modelo entienda que las 23:00 y las 00:00 son parte del mismo ciclo diario

        # Toma la columna Time_h (tiempo en horas) y la divide entre 24, usando una división entera (//), que redondea hacia abajo al número entero más cercano.
        # Esto convierte el tiempo en días completos. Por ejemplo:
        # Si Time_h = 25, entonces 25 // 24 = 1 (primer día).
        # Si Time_h = 50, entonces 50 // 24 = 2 (segundo día).
        # Si Time_h = 72, entonces 72 // 24 = 3 (tercer día).

        df['dia_actual'] = df['Time_h'] // 24
        
        # ------------------------------------------------------
        # INTERACCIONES AMBIENTALES
        # ------------------------------------------------------
       
        # En machine learning, a veces las variables individuales (como efecto_temp o efecto_pH)
        # no son suficientes para que el modelo entienda cómo interactúan entre sí.
        # Multiplicar variables crea características derivadas que capturan interacciones
        # entre factores, lo que puede mejorar las predicciones

        # la primera interacción es entre la temperatura, el pH y la eficiencia de la luz
        
        df['capacidad_fotosintetica'] = df['efecto_temp'] * df['efecto_pH'] * df['eficiencia_luminica__jassby_platt']
        
        # Esta interacciones capturan cómo la luz, la temperatura, la fotoinhibicion, el ph trabajan juntos para influir en la biomasa

        df['calidad_ambiental'] = df['capacidad_fotosintetica'] * (1 - df['fotoinhibicion'])
        

                              

        # ----------------------------------------------------
        # EFECTOS AVANZADOS DE CULTIVO: 
        # ----------------------------------------------------

       # if 'Culture_Age_h' in df.columns:

            # 1. Efecto de la densidad del cultivo
            # -------------------------------------
            # D = 1 / (1 + α * t)
            # -------------------------------------
            # - t es la edad del cultivo en horas (Culture_Age_h)
            # - α = 0.1 es el factor de atenuación
            #
            # La función modela cómo el cultivo se vuelve más denso con el tiempo:
            # - Comienza cerca de 1 (cultivo joven, poca densidad)
            # - Decrece gradualmente conforme envejece el cultivo
            # - Tiende asintóticamente a 0 en cultivos muy viejos
            #
            # Referencias:
            # - Molina Grima, E., et al. (1994). Effect of growth rate on the eicosapentaenoic acid 
            #   and docosahexaenoic acid content of Isochrysis galbana in chemostat culture.

           # df['efecto_densidad_cultivo'] = 1 / (1 + 0.1 * df['Culture_Age_h'])

            # 2. Penetración de luz efectiva
            # -----------------------------
            # P = E * D
            # -----------------------------
            # - E es la eficiencia lumínica base
            # - D es el efecto de la densidad
            #
            # Combina la eficiencia lumínica con el efecto de la densidad:
            # - Considera que la luz disponible disminuye en cultivos densos
            # - Modela el "self-shading" (auto-sombreado) del cultivo
            #
            # Referencias:
            # - Acién Fernández, F.G., et al. (1997). A model for light distribution and average
            #   solar irradiance inside outdoor tubular photobioreactors for microalgal mass culture
           
           # df['penetracion_luminica'] = df['eficiencia_luminica'] * df['efecto_densidad_cultivo']
        
        # ------------------------------------------------------
        # ENCODING DE FASES DE CRECIMIENTO
        # ------------------------------------------------------

        # Parto de la columna Growth_Phase de mi dataset. Growth_phase indica la fase de crecimiento del cultivo
        # (por ejemplo, "decline", "linear", "stationary") y transformo cada fase en columnas numéricas
        # aplicando one-hot encoding, que convierte cada categoría en una columna separada.
        # los modelos de machine learning no pueden trabajar directamente con texto (como "linear"), pero sí con números

        # pd.get_dummies crea columnas separadas para cada fase de crecimiento
        # Cada nueva columna tiene un valor de 1 si la fila corresponde a esa fase, o 0 si no
        # get_dummies es una función de pandas que convierte variables categóricas en variables dummy (0 o 1)
        
        if 'Growth_Phase' in df.columns:
            phase_dummies = pd.get_dummies(df['Growth_Phase'], prefix='phase')
        # pd.concat combina el DataFrame original con las nuevas columnas de fases, las columnas se añaden horizontalmente
            df = pd.concat([df, phase_dummies], axis=1)
        
        #--------------------------------------------------------
        # LIMPIAR DATOS NO VALIDOS
        #--------------------------------------------------------

        # Reemplazar infinitos y NaN
        # Reemplazo valores infinitos y NaN con el método forward fill (ffill) para rellenar hacia adelante
        # df.replace([np.inf, -np.inf], np.nan) reemplaza los valores infinitos por NaN
        # fillna(method='ffill') rellena los NaN con el último valor válido hacia adelante
        # fillna(0) rellena los NaN restantes con 0

        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        #---------------------------------------------------------
        # CLIP OUTLIERS
        # ---------------------------------------------------------

        # Clipping para manejar outliers
        # El código identifica y limita los valores extremos (outliers) en todas las columnas 
        # numéricas del DataFrame (df), excepto en la columna Biomass_g_L (que es el objetivo a 
        # predecir). Los outliers son valores inusualmente altos o bajos que pueden distorsionar
        # los modelos de machine learning. Utilizo una técnica llamada clipping para 
        # asegurarse de que todos los valores estén dentro de un rango aceptable

        # El clipping es una técnica que limita los valores extremos a un rango específico
        # En este caso, se usa el percentil 1 y 99 para definir los límites

        # df.select_dtypes(include=[np.number]) selecciona todas las columnas numéricas del DataFrame
        # y excluye las columnas no numéricas

        cols_numericas = df.select_dtypes(include=[np.number]).columns

        #Itera sobre cada columna numérica y aplica clipping a todas menos a 'Biomass_g_L' que es la variable a predecir
        # Esto es importante porque quiero poder conservar los valores reales de la biomasa

        for col in cols_numericas:

            # Si la columna no es Biomass_g_L, aplica clipping

            if col != 'Biomass_g_L':

                # Calcula los cuantiles 1% (Q1) y 99% (Q3) de la columna.
                # Q1 = df[col].quantile(0.01)
                # Q3 = df[col].quantile(0.99)
                # Esto define un rango que incluye el 98% central de los datos, considerando el 1% más bajo y el 1% más alto como outliers
                # pues solo me interesa quitar los valores extremos que puedan afectar al modelo
                
                Q1, Q3 = df[col].quantile([0.005, 0.995])

                # df[col].clip(Q1, Q3) limita los valores de la columna al rango [Q1, Q3]

                df[col] = df[col].clip(Q1, Q3)

        print(f"✅ Features Creados: {len(df.columns)} ")
        return df
    
# ========================================================================
# PASO 3: ENSEMBLE DE MÉTODOS DE SELECCIÓN DE CARACTERÍSTICAS
# ========================================================================

# La combinación de tres métodos diferentes (Correlación + SelectKBest + Random Forest)
# proporciona una selección de características más robusta y confiable porque:
#
# 1. CADA MÉTODO DETECTA PATRONES DIFERENTES:
#    - Correlación de Pearson: Detecta relaciones lineales directas
#    - SelectKBest (f_regression): Identifica relaciones estadísticas más complejas
#    - Random Forest: Captura interacciones no lineales y efectos combinados
#
# 2. COMPENSACIÓN DE DEBILIDADES:
#    - Correlación: Puede perder relaciones no lineales importantes
#    - SelectKBest: Puede pasar por alto interacciones complejas
#    - Random Forest: Puede ser sensible al ruido en los datos
#
# 3. VALIDACIÓN CRUZADA IMPLÍCITA:
#    - Si una variable es seleccionada por múltiples métodos
#    - Mayor confianza en su importancia real, pues se estaría seleccionando 
#       por su correlacion lineal con la biomasa, su importancia estadistica y su relevancia en un modelo de Random Forest
#    - Reduce el riesgo de sobreajuste a un único método
#    - Reduce la probabilidad de seleccionar features por casualidad
#
# 4. MAYOR ROBUSTEZ:
#    - Sistema más resiliente a fallos
#    - Si un método falla, los otros dos pueden compensar
#    - Mejor generalización del modelo final 

# # referencia:
# Brownlee, J. (2020). How to Choose a Feature Selection Method For Machine Learning
# Jason Brownlee explica que existen tres grandes categorías de técnicas de selección de características:
# Filter methods como la correlación o SelectKBest, que evalúan cada variable respecto al objetivo de forma independiente.
# Wrapper methods y embedded methods como Random Forest, que utilizan algoritmos predictivos para evaluar la contribución de cada característica
# a mayores,
# El artículo de D. Huang et al. (2023) introduce un método de selección de características basado en ensemble learning, 
# combinando varias técnicas mediante ponderaciones optimizadas (validación cruzada), 
# mejorando la precisión y robustez en modelos de predicción en series temporales

    def seleccion_y_preparacion_features(self, df, max_features=None):

        # Excluyo columnas que no son relevantes para la predicción de biomasa
        # Estas columnas seran importantes mas adelante para configurar la PINN y el LSTM
        
        _excluye_ = ['Biomass_g_L', 'Scenario', 'Growth_Phase', 'DateTime']
        
        # columnas_dataset crea una lista (columnas_dataset) con todas las columnas del DataFrame que no están en la lista excluye
        # [c for c in df.columns if c not in excluye] recorre todas las columnas del DataFrame df y excluye las que están en excluye
        # df[c].dtype in [np.number] asegura que solo se seleccionen columnas numéricas 

        columnas_dataset = [c for c in df.columns if c not in _excluye_ and df[c].dtype in [np.number]]
        
        # FILTRAR SOLO COLUMNAS NUMÉRICAS
        # df[columnas_dataset].select_dtypes(include=[np.number]) selecciona solo las columnas numéricas del DataFrame df
        # guardo las columnas numéricas en X_all

        X_all = df[columnas_dataset]

        # Aseguro que Biomass_g_L es la variable objetivo

        y = df['Biomass_g_L']
        
        # ------------------------------------------------------------------------
        # CORRELACION DE PEARSON
        # ------------------------------------------------------------------------

        # La correlación (correlación de Pearson) mide la relación lineal entre cada columna de mi dataframe 
        # (con los featues añadidos) y la variable objetivo (biomasa)

        # Un valor cercano a 1 o -1 indica una relación fuerte, mientras que un valor cercano a 0
        #  indica poca relación.
        
        # X_all.corrwith(y).abs() calcula la correlación absoluta para priorizar características
        # con relaciones fuertes, independientemente de si son positivas o negativas

        # sort_values(ascending=False) ordena las características por su correlación absoluta, de mayor a menor
        
        # esto me permite establecer un primer filtro para poder seleccionar las características
        # más relevantes que puedan influir a la hora de la prediccion de la biomasa
        
        correlations = X_all.corrwith(y).abs().sort_values(ascending=False)

        # correlaciones.head(n_features_max) selecciona las primeras 20 características con mayor correlación
        # index.tolist() convierte los índices de las correlaciones en una lista

        correlation_threshold = correlations.median()  # 0.2908
        top_corr = correlations[correlations > correlation_threshold].index.tolist()



        # X_all.columns.tolist()[:n_features_max] obtiene una lista de todas las columnas de X_all
        # en caso de error, selecciona las primeras 20 columnas

        # ------------------------------------------------------------------------
        # SCIKIT LEARN (SelectKBest)
        # ------------------------------------------------------------------------

        # SelectKBest es una clase con la que se puede seleccionar las mejores características
        # de un conjunto de datos para usar en el entrenamiento de modelos de aprendizaje 
        # supervisado basado en su importancia estadística. 
        # 
        # Evalúa la importancia de cada variable para predecir la biomasa
        # Para ello se le debe indicar una métrica de rendimiento, en mi caso f_regresion, con la que calcula
        # la correlacion entre cada característica y la biomasa. Después, convierte la correlación en un F-score. 
        # F-score es una metrica estadistica que mide la significancia de la relacion entre la varaible y la biomasa.
        #  Mayor F-score = Mayor importancia estadística.

        # Una vez obtenida la puntuación selecciona únicamente la K mejores del conjunto de datos.
        # Así, para poder usar esta clase, es necesario seleccionar dos hiperparámetros:
        #  la función de métrica (f_regression) y el valor de K = max_features como número de características a seleccionar

        # LA puntuacion de F-score no se mide como la de pearson (-1 a 1),
        # sino que es un valor positivo, cuanto mayor sea el F-score, más relevante es
        #    - F-score = 0: Variable no importante
        #    - F-score bajo (1-10): Importancia débil
        #    - F-score medio (10-100): Importancia moderada
        #    - F-score alto (>100): Importancia fuerte

        try:

            # f_regression es una función de Scikit-learn que calcula la correlación entre 
            # cada característica y la variable objetivo

            # Se usa para seleccionar las características más relevantes para la regresión
            # Crea un objeto SelectKBest de Scikit-learn, una herramienta para seleccionar las k mejores 
            # características según la métrica f_regression).

            selector = SelectKBest(f_regression, k='all')
            

            # Ajusta el modelo SelectKBest a los datos, x_all es el dataframe con todas las características e 'y' es la variable objetivo biomasa
            
            
            selector.fit(X_all, y)
            f_scores = selector.scores_
            f_threshold = np.median(f_scores)  # Mediana de F-scores
            selected_mask = f_scores > f_threshold
            
            # selector.get_support() devuelve un booleano indicando qué columnas 
            # fueron seleccionadas (las k con los puntajes F más altos).

            # X_all.columns[...] extrae los nombres de estas columnas.
            # .tolist() convierte los nombres en una lista (top_f), por ejemplo, ['Cell_Density_10E6_mL', 'temp_pH_synergy', 'pH']
          
            top_stats = X_all.columns[selected_mask].tolist()

            # top_stats contiene la lista de las mejores características seleccionadas por SelectKBest

        except:
            # En caso de error, se usa la lista de correlaciones como respaldo
            top_stats = top_corr
       
        # Referencias:
        # - Guyon, I., & Elisseeff, A. (2003). An introduction to variable and
        #   feature selection. Journal of Machine Learning Research, 3, 1157-1182.

        # ========================================================================
        # RANDOM FOREST
        # ========================================================================
        
        # En esta sección uso Random Forest para identificar las variables más 
        # relevantes para predecir la biomasa, complementando los métodos
        # de correlación y SelectKBest

        # Random Forest utiliza un ensemble de 50 (arbitrario) árboles de decisión para identificar las variables
        # más relevantes para predecir la biomasa. Cada árbol analiza diferentes
        # subconjuntos de datos y características, proporcionando una medida robusta de importancia
      

        # Cada arbol de decision tiene un nodo raiz inicial.
        # El nodo raiz contiene muestra aleatoria de datos de entrenamiento y es el punto de partida para construir cada árbol
        # Un SPLIT (division) es el proceso de dividir los datos en un nodo de un árbol en dos o más subnodos, 
        # basándose en una única característica (feature) y un valor de corte (threshold). 
        # El objetivo del split es crear subgrupos de datos más homogéneos con respecto a la variable objetivo

        # La idea de las divisiones y la creación de subnodos es poder llegar a obtener una división de los datos en subgrupos de la forma más eficiente posible,
        # de manera que cada subgrupo sea lo más homogéneo posible en relación a la variable objetivo (Biomass_g_L)

        # una vez conseguido esto se puede calcular la importancia de cada característica
        # en función de cuánto reducen los splits, cada split usa una única característica (feature), el error del modelo
        # es decir, la importancia de cada feature no se determina solo por la cantidad de splits sino por la calidad (cuanto reducen el error en la prediccion)

        # de otra forma:
        #  El algoritmo evalúa un subconjunto aleatorio de características en cada nodo para encontrar el mejor split
        # La importancia de cada característica se calcula tras construir el árbol, según cuánto contribuyen los splits
        # que la involucran a reducir el error.

        # 1. SENSIBILIDAD AL RUIDO:
        #    - Cada árbol se entrena con una muestra aleatoria de datos (bootstrap)
        #    - Si hay ruido en los datos, algunos árboles pueden aprender de este ruido,
        #     pues pueden llegar a crear splits muy específicos que se ajusten a los patrones aleatorios o ruido.
        #    - En datos biológicos, el ruido puede venir de:
        #      * Errores de medición en sensores
        #      * Variabilidad natural en el crecimiento
        #      * Fluctuaciones ambientales no controladas
        
        # 2. IMPORTANCIA DE CARACTERÍSTICAS:
        #    - Calcula importancia basada en cuánto mejora cada split
        #    - Variables ruidosas pueden parecer importantes por casualidad
        #    - Necesita múltiples árboles para promediar y reducir este efecto, 50 está bien para esto
        
        # 3. COMPENSACIÓN:
        #    - Usar 50 árboles ayuda a reducir el impacto del ruido
        #    - Cada árbol ve una muestra diferente de datos
        #    - El promedio de muchos árboles es más robusto
        
        # Por esto es importante combinarlo con otros métodos (correlación y SelectKBest)
        # que son menos sensibles al ruido.
       
        try:
             # 1. Crear y entrenar modelo Random Forest
            # ---------------------------------------
            # - n_estimators=50: usa 50 árboles de decisión (es eficiente y relativamente rápido, más de 50 da resultados parecidos pero con mayor procesamiento y tiempo)
            # - random_state=42: para reproducibilidad de resultados
           
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_all, y)
            
            # 2. Extraer importancia de características
            # ---------------------------------------
            # Crea una Serie de pandas donde:
            # - rf.feature_importances_: array con la importancia de cada variable
            # - index=X_all.columns: nombres de las variables como índices
            
            
            importances = rf.feature_importances_
            importance_threshold = np.median(importances)  # Mediana de importancias

            # 3. Seleccionar top features
            # ---------------------------------------
             # - nlargest(): selecciona las max_features variables más importantes
             # - index.tolist(): convierte los nombres de variables en lista
            top_rf = X_all.columns[importances > importance_threshold].tolist()

        except:
            # Si falla el Random Forest, usa las correlaciones como plan B
            # Esto asegura que el sistema siga funcionando incluso si hay errores
            top_rf = top_corr
        
        # ========================================================================
        # COMBINAR MÉTODOS (CORRELACIÓN + SELECTKBEST + RANDOM FOREST)
        # ========================================================================

        # Aquí combino las características seleccionadas por correlación, SelectKBest y Random Forest
        # la idea es que con esta sumatoria de metodos, concateno en una sola lista las mejores caracteristicas de las 3, hasta el límite max_features
        # combino los 3 métodos de selección de características (top_corr, top_stats y top_rf), 
        # aseguro el número mínimo de características, 
        # y creo el DataFrame final X para el entrenamiento
        
        self.selected_features = list(set(top_corr + top_stats + top_rf))[:max_features]
       
        # - set(): elimina duplicados
        # - [:max_features]: limita el número total de features

        # - X_all[self.selected_features]: selecciona las columnas del DataFrame original que están en self.selected_features
        X = X_all[self.selected_features]

        print(f"Variables en X_all: {len(X_all.columns)}")
        print(f"Variables en correlations: {len(correlations)}")
        print(f"max_features real: {max_features}")
        print(f"top_corr length: {len(top_corr)}")
        print(f"top_stats length: {len(top_stats)}")  
        print(f"top_rf length: {len(top_rf)}")
        print(f"Variables con correlación 0: {(correlations == 0).sum()}")

        print(f"🎯 Features seleccionadas {len(self.selected_features)} ")






        #Create a bar plot of feature importance (if available from Random Forest)
        if top_rf:
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_all, y)
            importance_df = pd.DataFrame({
            'Feature': X_all.columns,
            'Importance': rf.feature_importances_
        })
        importance_df = importance_df[importance_df['Feature'].isin(self.selected_features)]
        importance_df = importance_df.sort_values('Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        plt.xlabel('Importancia')
        plt.title('Importancia de las Features Seleccionadas (Random Forest)')
        plt.gca().invert_yaxis()  # Invert y-axis for better readability
        plt.tight_layout()
        plt.show()





        print(f"📊 Top 18: {self.selected_features[:18]}")
        
        # ------------------------------------------------------
        # DIVISIÓN DE DATOS (train-test split)  
        # ------------------------------------------------------
        
        # data leakage: fuga de datos 
        # el data leakage ocurre cuando el modelo tiene acceso a información que no debería tener durante el entrenamiento,
        # lo que puede llevar a una sobreestimación de su rendimiento.

        # en mi dataset, tengo una columna llamada Scenario, que va de 1 hasta 60, esta columna indica el cultivo al que pertenece cada fila
        # mi data set es muy completo porque contemplo 60 escenarios diferentes (cultivos), cada uno con sus propias condiciones ambientales y de cultivo
        # por eso, para evitar el data leakage, divido los datos en entrenamiento y prueba pero respetando el cultivo (Scenario) al que pertenecen.
        # de esta forma, el modelo solo ve datos de ciertos cultivos durante el entrenamiento y se prueba en cultivos diferentes que no ha visto antes
        #  75% para entrenamiento y 25% para testeo

        # con 25% de los cultivos para testeo o validación, me refiero a la validación final del modelo, es decir, guardaré estos datos para evaluar el rendimiento del modelo después de entrenarlo

        # del 75% que uso para entrenamiento, usaré una proporcion del 80-20 para entrenar pesos y para validar internamente durante el entrenamiento.

        # continúo:
        # yo necesito que la prediccion sea continua en el tiempo, es decir,
        # me interesa respetar la secuencia temporal para poder simular la realidad, 
        # pues uso datos pasados para predecir el futuro, por eso es importante asumir cada escenario como
        # un bloque continuo en el tiempo

        # me interesa agrupar los datos por escenarios, de lo contrario si mezclo todos los datos
        # y luego realizo el entrenamiento con los datos mezclados, el modelo podría aprender patrones
        # que no son representativos de la realidad, ya que los datos de entrenamiento y prueba
        # estarían mezclados y el modelo podría "ver" información futura durante el entrenamiento. 

        # referencia: 
        # Dado que los datos de crecimiento de las microalgas proceden de cultivos (escenarios) independientes con múltiples 
        # mediciones temporales por cultivo, estos no cumplen la suposición de ser independientes e idénticamente distribuidos, 
        # (Sasse et al. (2025)), en tales casos un split aleatorio entre filas puede provocar que muestras 
        # dependientes de un mismo cultivo queden repartidas en entrenamiento y prueba, generando data leakage y predicciones 
        # artificialmente elevadas. 
        # Por eso, se justifica separar los cultivos completos (es decir, los Scenario) entre los conjuntos de 
        # entrenamiento y evaluación para estimar correctamente la capacidad de generalización real del modelo

        # Sasse, L., et al. (2025). Overview of leakage scenarios in supervised machine learning. Journal of Big Data


        # Lo primero es verificar que tengo la columna 'Scenario' para división en bloques de cultivo
        # 'Scenario' es una columna que indica el escenario (cultivo) al que pertenece cada fila

        if 'Scenario' in df.columns:
            # escenarios disponibles (cultivos)
            scenarios = np.array(sorted(df['Scenario'].unique()))

            # sorted ordena los escenarios de menor a mayor
            # df['Scenario'].unique() obtiene una lista de los cultivos únicos (sin duplicados)
            # np.array convierte la lista en un array de numpy para facilitar operaciones posteriores
            # es decir, con esto obtengo una lista ordenada de los escenarios, ordeno los cultivos de menor a mayor, posiciono el número de cultivos de 1-60 de forma ordenada


        # ---------------------------------------------- 
        # REPRODUCIBILIDAD Y SELECCION ALEATORIA
        # ----------------------------------------------

        # La reproducibilidad es la capacidad de obtener resultados consistentes e idénticos 
        # cuando se repite un experimento bajo las mismas condiciones. En ciencia de datos 
        # y machine learning, significa que cualquier persona debe poder ejecutar el código y
        # obtener exactamente los mismos resultados.
        # Para garantizar la reproducibilidad, se establece una semilla aleatoria (np.random.seed(n))
        # Esto asegura que las operaciones aleatorias (como la división de datos) produzcan los mismos resultados en cada ejecución.
            
            SEED = 50                      # semilla fija para reproducibilidad
            np.random.seed(SEED)           # fija la semilla global para NumPy
            
            # np.random.seed() genera una secuencia de números pseudoaleatorios (a traves de un algoritmo)
            # con esta secuencia fija, las operaciones aleatorias producirán los mismos resultados en cada ejecución del código


        # La idea es que siempre se seleccionen 15 cultivos (de forma aleatoria), de los 60 totales, para validacion.
        # La clave de esto es que en cada ejecución del codigo, los 15 cultivos, se seleccionen de forma aleatoria

        # de esta forma, el modelo se entrena siempre con 45 cultivos y se valida con 15 cambiandose los cultivos de validacion en cada ejecucion, de manera aleatoria
        # esto ayuda a evitar sesgos y asegura que el modelo generalice bien a diferentes cultivos
        # es decir, que no se sobreajuste a un conjunto fijo de cultivos de validacion

        # ===================================================================================================================================
        # NOTA IMPORTANTE:
        # en la version inicial del código, usaba los primeros 45 cultivos para entrenamiento y los últimos 15 para validación, siempre.
        # Al implementar esta parte, la selección aleatoria de escenarios para validación, he mejorado notablemente la capacidad de mi modelo predictivo,
        # pasando de un R2 de 0.85 a un R2 de 0.93, lo que indica una mejor generalización y precisión en las predicciones

        # esta parte es muy  importante porque añade REPRESENTATIVIDAD: con selección aleatoria (aunque fijada por seed), 
        # el conjunto de validación suele ser más parecido al global, y el modelo rinde mucho, pero mucho mejor.
        # =====================================================================================================================================
        
            n_total = len(scenarios) # total de escenarios (60)
            n_val = max(1, n_total // 4)   # divido el total de escenarios entre 4 (25%), resultado 15 (mínimo 1 escenario)
            validacion_scenarios = np.random.choice(scenarios, size=n_val, replace=False)
            # np.random.choice es una función de NumPy que sirve para seleccionar elementos de un conjunto de datos de forma aleatoria,
            # en el fondo se rige por la semilla que he fijado antes (50), por lo que la selección será siempre la misma en cada ejecución del código
            # 15 cultivos seleccionados para validación de forma pseudoaleatoria pero siempre los mismos en cada ejecución del código.

            # estos escenarios se usarán para validación (test)

            
            entrenamiento_scenarios = ~df['Scenario'].isin(validacion_scenarios)
            # isin() devuelve un booleano indicando si cada fila pertenece a los escenarios de validación
            # ~ invierte el booleano, es decir, TRUE para filas que NO están en los últimos 15 escenarios (es decir, los otros 45 escenarios)
            # estos son los que se usarán para entrenamiento

            X_entrenamiento, X_validacion = X[entrenamiento_scenarios], X[~entrenamiento_scenarios]
            # X_entrenamiento representa las features (entradas o predictores) de las filas que pertenecen a los primeros 45 escenarios (entrenamiento)
            # X_validacion representa las features (entradas o predictores) de las filas que pertenecen a los últimos 15 escenarios
            Y_entrenamiento, Y_validacion = y[entrenamiento_scenarios], y[~entrenamiento_scenarios]
            # Y_entrenamiento representa la biomasa (objetivo) de las filas que pertenecen a los primeros 45 escenarios (entrenamiento)
            # Y_validacion representa la biomasa (objetivo) de las filas que pertenecen a los últimos 15 escenarios (prueba o validacion)
            
            print(f"🚂 Division de datos: Entrenamiento {len(X_entrenamiento)}, Validacion {len(X_validacion)}")


        else:
            X_entrenamiento, X_validacion, Y_entrenamiento, Y_validacion = train_test_split(X, y, test_size=0.25, random_state=42)
            # en caso de no tener la columna Scenario, hago un split aleatorio normal (25% validacion, 75% entrenamiento)
        
        #  ------------------------------------------------------
        # NORMALIZACIÓN DE DATOS (StandardScaler)
        #  ------------------------------------------------------

        # self.scalers es un diccionario que contiene los objetos de normalización
            # self.scalers['features'] se usa para normalizar las características (X_entrenamiento, X_validacion)
            # self.scalers = {
                    #'features': StandardScaler(),  # EScalador estandar las características X
                    #'target': StandardScaler()     # Escalador estandar ara la variable objetivo y
            #}
            # fit_transform ajusta el escalador a los datos de entrenamiento y transforma los datos al mismo tiempo
            
            
            # X_entrenamiento_scaled es un DataFrame que contiene las características de entrenamiento normalizadas
            # pd dataframe convierte el array de numpy devuelto por fit_transform en un DataFrame de pandas
            # con las mismas columnas y el mismo índice que X_entrenamiento
            # self.scalers['features'] es un objeto StandardScaler de Scikit-learn que normaliza las características
            # transform aplica la normalización a los datos de prueba (X_entrenamiento) 
            # colums= X_entrenamiento.columns asegura que las columnas del DataFrame resultante sean solo las de X_entrenamiento
            # index=X_entrenamiento.index asegura que el índice del DataFrame resultante sea el mismo que X_entrenamiento

        X_entrenamiento_scaled = pd.DataFrame(
            self.scalers['features'].fit_transform(X_entrenamiento), 
            columns=X_entrenamiento.columns, index=X_entrenamiento.index
        ) # hago lo mismo con los datos de validacion:
        X_validacion_scaled = pd.DataFrame(
            self.scalers['features'].transform(X_validacion), 
            columns=X_validacion.columns, index=X_validacion.index
        )
        # aqui si que es necesario trabajar con el historico de datos de la biomasa pues es necesario
        #  entrenar y validar los modelos con la variable que se quiere predecir


         # Normalizo la variable objetivo (y_train, y_test) usando el mismo escalador
            # fit_transform es un metodo de standerdscaler que ajusta los datos de entrenamiento de la variable objetivo
            # devuelve un array de numpy con los datos normalizados
            # y_train.values.reshape(-1, 1) convierte la serie de pandas en un array de numpy de una sola columna
            # (-1, 1) asegura que sea un array bidimensional con una sola columna
            # transform aplica la normalización a los datos de prueba (y_test)
            # flatten() convierte el array de numpy en un array unidimensional
            # esto significa que los datos de la variable objetivo se normalizan
            # Y_entrenamiento_scaled es un array de numpy con los datos normalizados de la variable objetivo
            # quedando por ejemplo: [0.1, 0.2, 0.3, ...]
            
        Y_entrenamiento_scaled = self.scalers['target'].fit_transform(Y_entrenamiento.values.reshape(-1, 1)).flatten()
        Y_validacion_scaled = self.scalers['target'].transform(Y_validacion.values.reshape(-1, 1)).flatten()
        
        return X_entrenamiento_scaled, X_validacion_scaled, Y_entrenamiento_scaled, Y_validacion_scaled, Y_entrenamiento, Y_validacion

# ===================================================================================================
# PASO 5 : SISTEMA MULTI-MODELO (modelos predictivos y algoritmo de combinación)
# ===================================================================================================
#    Modelos:
#       - PINN (Physics-Informed Neural Network)
#       - LSTM (Long Short-Term Memory)
#       - Regresión Lineal
#       - Ridge
#       - Random Forest
#       - XGBoost
#    Ensemble Ponderado
#       - Pesos basados en rendimiento
#       - Validación temporal

# ========================================================================
#  RED NEURONAL INFORMADA POR RESTRICCIONES BIOLÓGICAS (PINN)
# ========================================================================

# Esto es una red neuronal artificial de tipo perceptrón multicapa que incorpora
# conocimiento biológico en su función de pérdida para predecir la biomasa de microalgas
# en función de variables ambientales y de cultivo
#  
# La arquitectura se organiza en varias capas densas intercaladas con funciones de activación no lineales 
# y con técnicas de regularización mediante dropout, lo que permite capturar relaciones complejas entre las 
# variables de entrada y al mismo tiempo reducir el sobreajuste. 
# 
# La primera capa incluye una normalización por lotes, que garantiza la homogeneidad de las variables en cada 
# iteración de entrenamiento y facilita la convergencia del modelo.
# 
#  La salida de la red corresponde a un único valor continuo que representa la biomasa estimada en el cultivo.
#  La particularidad de este modelo está en la función de pérdida, que combina el error cuadrático medio con
#  un término de penalización aplicado cuando la predicción resulta negativa. 
# 
# Biológicamente no es posible una biomasa menor que cero, entonces, esta formulación introduce en el aprendizaje una restricción 
# coherente con el conocimiento de la fisica que regula a esta variable, convirtiendo al modelo en un ejemplo de red neuronal informada
# por la física o por la biología, lo que asegura que las predicciones sean no solo precisas, 
# sino también consistentes con la realidad del sistema estudiado.


# ------------------------------------
# FUNCIONAMIENTO DE LA RED NEURONAL
# ------------------------------------
# Es un modelo matematico que se construye con neuronas artificiales, pequeñas unidades de calculo
# que reciben datos de entradas, los combinan con unos pesos, suma un sesgo, y pasa el resultado por
# una función que decide cuánto de esa señal se transmite hacia adelante (capas).

# Con la unión de varias capas se forma la red neuronal. Cada capa transforma los datos un poco más y 
# al final, la salida es un número o una etiqueta: en este caso, la biomasa predicha.

# Cuantas más neuronas haya, mayor capacidad tendrá la red para aprender relaciones complejas, pero si pongo demasiadas
# es más probable que memorice los datos en lugar de generalizar. Por eso 64-32 es un término medio ampliamente utilizado 
# y es el que utilizo para este modelo.

# NORMALIZACIÓN POR LOTES:
# BatchNorm  es una función que normaliza cada feature (caracteristica) dentro de un lote de entrenamiento (batch). 
# genera un vector de features normalizados con media 0 y desviación 1. Una vez obtengo este vector ya noramlizado, 
# entra en la primera capa lineal Linear(input_size,64), donde se calculan las primeras combinaciones 


# ReLU: Rectified Linear Unit, es una función de activación que se utiliza en cada una de las capas,
# después de que cada neurona artificial combine variables con pesos.
# Esto es necesario para introducir la NO LINEALIDAD, permitiendo que la red aprenda realciones complejas 
# y no sólo combinaciones lineales con pesos.

# f(x)=max(0,x)

# esta funcion lo que hace es dejar pasar los valores positivos de X y convertir en 0 los valores negativos de X
# por ejemplo, aplicando la ecuacion a mi modelo, en la primera capa, una neurona aprende una combinación de las variables de esta forma:

# h = 0.5⋅* pH − 0.2⋅*nutrientes + 0.1⋅*tiempo

# Ese h puede salir negativo, por ejemplo si los nutrientes son muy altos y pH bajo.

# Con ReLU, si h=−3.2, la salida de la neurona será 0.
# Si h=2.5, la salida será 2.5

# Así, ReLU actúa como un filtro: ignora combinaciones no útiles (cuando salen negativas) y deja pasar las útiles.

# En cada capa, durante el entrenamiento se produce además una técnica de regularización llamada dropout,
# que lo que hace es apagar aleatoriamente un porcentaje de neuronas en una capa.

# Dropout(0.3), significa que en cada paso de entrenamiento, en promedio el 30% de esas neuronas no participan, 
# y solo el 70% restante sigue activo.

# El objetivo del dropout es evitar el sobreajuste, obligando a que el aprendizaje se distribuya entre todas y no dependa únicamente 
# de unas pocas neuronas concretas que den siempre la respuesta. 

# Es importante porque la red debe aprender patrones robustos que no dependan de unas pocas neuronas en específico, 
# sino que todas tienen que poder aportar algo.  Así, nunca se sabe cuáles estarán apagadas en la siguiente iteración, 
# forzando a que la red siga funcionando bien aunque varias neuronas no estén disponibles.

# Repartir el conocimiento aporta una representación mucho más robusta del cultivo.


class CompactPINN(nn.Module): # Primero declaro una clase de red neuronal tipo MLP informada por restricciones biológicas
   
    def __init__(self, input_size): # Constructor: recibe el número de características de entrada (columnas/features)
        super().__init__()          # Inicializo la superclase nn.Module
        self.net = nn.Sequential(   # Defino la arquitectura como una secuencia de capas
            nn.BatchNorm1d(input_size), # Normaliza por lotes cada feature para estabilizar y acelerar el entrenamiento
            nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(0.3), # Capa densa a 64 neuronas + ReLU + Dropout 30% para regularizar
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), # Segunda capa densa a 32 neuronas + ReLU + Dropout 20%
            nn.Linear(32, 1)                               # Capa de salida escalar: predice biomasa (valor continuo)
        )
    

    # ---------------------------------------------------------
    # FUNCION FORWARD
    # ---------------------------------------------------------
    # En PyTorch, una red neuronal define un método forward(x).
    # Aquí es donde se describe el procedimiento que hacen los datos
    # a través de las capas del modelo hasta producir una salida.
    #
    # En este caso, self.net contiene toda la secuencia de capas:
    #   1. BatchNorm → normaliza los valores de entrada (pH, nutrientes, tiempo, etc.)
    #   2. Linear + ReLU + Dropout → combinaciones no lineales y regularización
    #   3. Linear + ReLU + Dropout → segunda transformación intermedia
    #   4. Linear final → genera una predicción escalar (biomasa)
    #
    # Al llamar al modelo con un lote de datos X (ejemplo: 32 filas x 20 features),
    # este forward se encarga de pasarlos por todas esas capas en orden.
    #
    # El resultado final tiene forma [batch_size, 1], es decir, una predicción de biomasa
    # para cada fila del lote de entrada.

    def forward(self, x): # x es un tensor de PyTorch con las características de entrada
        return self.net(x)  # Pasa los datos de entrada x a través de la red definida en self.net
    

     # ------------------------------------
     # Funcion de perdida personalizada
     # ------------------------------------

     # El objetivo de esta función es calcular cuánto se equivoca 
     # la red en sus predicciones e imponer una restricción biológica:
     # la biomasa no puede ser negativa.

     # mse = nn.MSELoss()(pred.squeeze(), target)

     # Calcula el error cuadrático medio entre las predicciones del modelo (pred) y los 
     # valores reales de biomasa (target).
     # "squeeze()" elimina dimensiones sobrantes, establece la misma dimension de vector tanto para pred como para target.

     #  bio_penalty = torch.mean(torch.relu(-pred)) * 5

     # Aplico ReLU a (-pred), convirtiendo en positivos los valores de predicción que sean negativos,
     # luego calculo la media de estos "errores biologicos" y por último multiplico x5 para auumentar la penalización.

     # return mse + 0.1 * bio_penalty

     # El error total es el error cuadrático medio sumado a una penalización del 10% cuando la red intenta predecir valores de biomasa negativos.

     # ***nota: ********
     # se podría mejorar esta parte, cambiando la fijación manual de penalización (5 para la dureza de la penalización y 0.1 para el peso relativo) 
     # por un sistema de pesos adaptativos en función de cuánto se equivoque en la prediccion,
     # para que el sistema aprenda la ponderación entre MSE y penalización durante el entrenamiento. Un multiplicador de Lagrange podría estar bien.

     # justifico un x5 en dureza de la penalización y un 0.1 para su peso relativo, porque al probar con otros valores 
     # (2 10 15) o (0.1 0.3 0.4) el modelo es un poco peor.


    def bio_loss(self, pred, target): 
        mse = nn.MSELoss()(pred.squeeze(), target)
        bio_penalty = torch.mean(torch.relu(-pred)) * 5  # Penalización por predicciones negativas
        return mse + 0.1 * bio_penalty # Peso relativo de la penalización sumada al MSE

# =======================================================================
# RED NEURONAL LSTM (Long Short-Term Memory)
# =======================================================================

# CompactLSTM es una clase que define una red neuronal basada en LTSM (Long Short-Term Memory)
# Esta red está diseñada para capturar dependencias temporales en los datos (series temporales).
# De esta forma, la idea es que el modelo pueda "recordar" cómo evoluciona la biomasa en el tiempo en función de las condiciones del cultivo.
# Debe aprovechar la información de los pasos previos para mejorar la prediccón en el momento actual.

# Importante: La LSTM ya está diseñada para manejar series temporales, yo sólo le doy la secuencia ordenada en el tiempo,
# y ella se encarga de recordar cómo estaban las variables en pasos anteriores y combinarlas con el presente.

# definicion de la capa LSTM:

# self.lstm = nn.LSTM(input_size, 32, batch_first=True, dropout=0.2)

# La LSTM recibe como entrada secuencias con el tamaño (input_size) 
# Se configuran 32 unidades ocultas (hidden units), una unidad  oculta es un neurión artificial donde se realizan todas 
# las operaciones en cada capa, con esto defino la dimensionalidad interna.

# batch_first=True significa que los tensores de entrada a la LSTM se organizan como [número de muestras, longitud de la secuencia, número de variables]
# esto es lo mas comun a la hora de crear una LSTM.

# Se aplica un dropout=0.2 para mejorar la generalización y reducir el riesgo de sobreajuste.


class CompactLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()         # Inicializo la superclase nn.Module, que es la base de todas las redes neuronales en PyTorch
        self.lstm = nn.LSTM(input_size, 32, batch_first=True, dropout=0.2) # Capa LSTM con 32 unidades ocultas y dropout del 20%
       
       
       # Definición de la red de salida o capa densa (fully connected)
        # ---------------------------------------------------------------
        # La salida de la LSTM en el último paso temporal (dimensión 32)
        # se procesa por una pequeña red densa:
        # - Linear(32 -> 16): capa oculta intermedia con 16 neuronas
        # - ReLU(): función de activación no lineal que introduce capacidad
        #   de aprendizaje de relaciones complejas
        # - Dropout(0.2): regularización adicional
        # - Linear(16 -> 1): capa final que predice la biomasa (un valor escalar)

        self.out = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 1)) # Capas densas para procesar la salida de la LSTM y generar la predicción final
   
    # forward funciona igual que en el caso anterior

    def forward(self, x): # x es un tensor de PyTorch con las características de entrada
        # x tiene forma [batch_size, sequence_length, input_size].
        # La LSTM procesa toda la secuencia y devuelve:
        # - lstm_out: representaciones ocultas de cada paso temporal
        # - _: (hidden_state, cell_state) que no se utilizan aquí
        lstm_out, _ = self.lstm(x) # Paso los datos de entrada x a través de la capa LSTM

        return self.out(lstm_out[:, -1, :]) # Uso solo la salida del último paso temporal para la predicción final
        # Selecciono la salida correspondiente al último paso temporal
        # (lstm_out[:, -1, :]) ya que es la que resume toda la secuencia previa.
        # Esto es porque, en el último paso, la LSTM ha procesado toda la secuencia y su vector oculto contiene 
        # una codificación comprimida de la historia completa.

        # Esta representación comprimida se pasa a la red fully connected (capa densa), produciendo la predicción final de biomasa.

# ========================================================================
# MODELO COMBINADO(combina varios modelos y pondera por rendimiento)
# ========================================================================

# El modelo combinado se basa en una idea sencilla: combinar varios modelos diferentes para obtener un resultado final 
# más preciso y estable que cualquiera de ellos por separado.

# La idea es que cada modelo tiene sus propias fortalezas y debilidades, y al combinarlos,
# se pueden compensar los errores individuales y aprovechar las ventajas de cada uno.

# He contemplado 3 enfoques diferentes:
# - modelos lineales (Linear y Ridge)
# - modelos de árboles de decisión (Random Forest y XGBoost)
# - redes neuronales (PINN y LSTM)

# Cada uno de estos enfoques tiene características distintas que pueden ser útiles para predecir la biomasa:
# Los modelos lineales sirven para encontrar tendencias generales, relaciones lineales, como por ejemplo si al aumentar los nutrientes
#  también aumenta la biomasa de forma proporcional.

# Los modelos de arbol de decision no buscan una relación directa o proporcional sino que dividen los datos en distintas condiciones,
# permitiendo descubrir interacciones más complejas, como que el efecto de los nutrientes en la biomasa depende de un rango específico de pH 
# (explicado en la sección de RandomForest)

# XGBoost es parecido pero con un enfoque más avanzado: va corrigiendo los errores de otros árboles anteriores y ajustándose cada vez más a los datos.

# Con la PINN le "enseño" al modelo que no puede predecir valores negativos de biomasa, porque eso no tiene un sentido real.
# Y la LSTM, está pensada para trabajar con información que cambia en el tiempo, para que pueda aprender de cómo han ido cambiando los cultivos a lo largo del tiempo y tener en cuenta esta evolución.

# una vez definidos los modelos, se entrenan todos con los mismos datos de entrenamiento.
# aplicando validación cruzada interna (hold-out), utilizo una proporción 80-20 dentro del bloque de netrenamiento.
# es decir, de los 45 cultivos que uso para entrenamiento,  36 cultivos son para entrenar pesos y 9 cultivos para validar internamente.

# criterio adaptativo: antes de hacer la combinación de todos los modelos, cada modelo se evalúa en un conjunto de validación (los 15 cultivos restantes), 
# que son datos que no han visto durante el entrenamiento. En ese conjunto se mide su error con el error cuadrático medio (MSE). 

# Calculo el error (MSE) de cada modelo, y calculo su inverso:

# calculo el inverso (1/Ei) del error, obtengo los valores proporcionales a la calidad de cada modelo cuanto menor es el error, mayor es el valor.
# el problema aqui es que si utilizo los valores resultantes directamente como pesos, el resultado final no estaría en una escala definida y no habría forma de interpretar correctamente qué aporta cada modelo realmente.

# Por eso realizo una normalización, que consiste en dividir cada valor inverso por la suma total de los inversos:

# peso_n = (1/Ei) / ( ∑_{i=1}^{M} 1/Ei)

# Así, saco los pesos reales normalizados de cada modelo:

# Pred_final = (peso_1 * pred_A) + (peso_2 * pred_B) + ... + (peso_n * pred_m)

# De esta forma consigo que mi modelo combinado se asegure de que los mejores modelos tengan más voz en la decisión y
# que los peores apenas influyan.

class CompactMultiModel:
    def __init__(self):
        # Diccionario para almacenar los modelos entrenados
        self.models = {}
        self.val_scores = {}
        
    def train_all(self, X_entrenamiento, Y_entrenamiento, epochs=150):
        # Función para entrenar todos los modelos y calcular sus pesos basados en el rendimiento de validación
        print("\n🚀 Entrenando modelos")
        
        # ------------------------------------------------------------------
        # 1) VALIDACIÓN INTERNA (hold-out): 80% train / 20% validación
        # ------------------------------------------------------------------

        # - Este split es *solo* dentro del bloque de ENTRENAMIENTO (es decir,
        #   NO toca los 15 escenarios de validación final que guardé aparte).
        # - Sirve para: (a) ajustar las RNA (paradas tempranas sencillas) y
        #   (b) medir el error de cada modelo y derivar pesos del ensemble.
        
        X_tr, X_val, y_tr, y_val = train_test_split(X_entrenamiento, Y_entrenamiento, test_size=0.2, random_state=42)

        # test_size=0.2 significa que el 20% de los datos se usan para validación interna
        # random_state=42 fija la semilla para que la división sea reproducible
        
        # ------------------------------------------------------------------
        # 2) ENTRENAMIENTO DE MODELOS 
        # ------------------------------------------------------------------
        # Aquí defino y entreno varios modelos base con diferentes enfoques:    
        #   * Lineal/Ridge: relaciones proporcionales y efecto de regularización.
        #   * RandomForest: no lineal, maneja interacciones y umbrales.

        self.models['Linear'] = LinearRegression().fit(X_tr, y_tr)

        # -----------------------------------------------------
        # -----------------------------------------------------
        # Regularizacion L2 en Ridge Regression
        # -----------------------------------------------------
        # -----------------------------------------------------
        # *Regresion Ridge:
        # A la hora de entrenar un modelo, lo que se hace es ajustar los coeficientes para minimizar el error entre lo que se predice y lo real.
        # Uno de los problemas que pueden aparecer es, si mis datos tienen muchas variables o incluso se correlacionan mucho entre sí, que los coeficientes 
        # podrían llegar a crecer de forma desproporcionada. Esto produce que el modelo se sobreajuste: funciona bien en entrenamiento pero
        # muy mal en validación o con nuevos datos.

        # La regularización lo que hace es añadir un término extra de 'penalización' a la función de coste, para castigas los modelos demasiado complejos
        # En el caso de Ridge Regression (y AdamW más adelante) incorporan una regularización L2, cuya particularidad es que,
        # en este caso (regularizacion L2), el castigo o la penalización es la suma de los cuadrados de los coeficientes.
        # Dicho de otra forma, es un moderador, o regualdor, de pesos específico que mitiga el overfitting.

        self.models['Ridge'] = Ridge(alpha=1.0).fit(X_tr, y_tr)

        # *RandomForest:
        # RandomForest es un conjunto de árboles de decisión que dividen los datos en función de condiciones específicas.
        # Por ejemplo, un árbol podría aprender que si el pH > 7.5 y los nutrientes > 50, entonces la biomasa es alta.
        # Otro árbol podría aprender que si el pH < 6.5 y la temperatura < 20°C, entonces la biomasa es baja.
        # Al combinar muchos árboles (100 en este caso), el modelo puede capturar interacciones complejas entre las variables.
        # max_depth=8 limita la profundidad de cada árbol para evitar sobreajuste.
        # random_state=42 fija la semilla para reproducibilidad.
        self.models['RandomForest'] = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42).fit(X_tr, y_tr)
    
        
        # *XGBoost:
        # XGBoost es una implementación avanzada de boosting que crea árboles secuenciales,
        # donde cada nuevo árbol intenta corregir los errores de los anteriores.
        # Esto permite capturar patrones complejos y mejorar la precisión.
        # Parámetros como learning_rate=0.03 controlan la velocidad de aprendizaje,
        # subsample=0.8 y colsample_bytree=0.8 introducen aleatoriedad para mejorar la generalización.
        # reg_alpha y reg_lambda son términos de regularización para evitar sobreajuste.
        # random_state=42 asegura reproducibilidad.
        
        self.models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42
        ).fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        # ------------------------------------------------------------------
        # 4) Redes neuronales: PINN y LSTM
        # ------------------------------------------------------------------
        # Las redes en PyTorch trabajan con tensores (no con DataFrames de pandas).
        # Por eso tengo que convertir X (features) e y (objetivo) desde pandas → tensores float32.
        X_tr_t, y_tr_t = torch.FloatTensor(X_tr.values), torch.FloatTensor(y_tr) # X_tr_t = [n_trains,n_features] frente a y_tr_t = [n_trains]
        X_val_t, y_val_t = torch.FloatTensor(X_val.values), torch.FloatTensor(y_val)
        
        # ---------------------------------
        # PINN
        # ---------------------------------

        # creo una instancia del modelo PINN definido antes.
        # X_entrenamiento.shape[1] devuelve el nº de variables de entrada (features)

        # el nº de features pasa al constructor del modelo para definir el tamaño de la capa de entrada:

        #  self.models['PINN'] = CompactPINN(X_entrenamiento.shape[1])

        # Ahora defino el optimizador para entrenar la PINN.:
        # un optimizador es un método de descenso por gradiente que ajusta los pesos de la red para que las predicciones se acerquen lo máximo posible a los datos reales.
        # Adam es un optimizador tiene la característica de que se adapta automáticamente a cada peso de la red calculando promedios del gradiente pasado y del cuadrado del gradiente.
        # Es mejor que otros optimizadores básicos como SGD, pero a día de hoy se utiliza un optimizador que funciona mucho mejor, llamado AdamW.

        # AdamW corrige varios errores de la versión Adam original y se ha convertido en el estándar en deep learning.
        # Con AdamW consigo que la red neuronal aprenda rápido, en un sistema con muchas features, que sea estable y que tenga regularización L2 para evitar el sobreajuste.
        # Así, utilizo: AdamW con lr=0.001 y weight_decay=0.01 (regularizacion L2), que son valores estándar y seguros.

        self.models['PINN'] = CompactPINN(X_entrenamiento.shape[1])
        opt_pinn = torch.optim.AdamW(self.models['PINN'].parameters(), lr=0.001, weight_decay=0.01)
        

        # Guardo el mejor error de validación encontrado hasta ahora.
        # float('inf') es "infinito positivo", un valor inicial muy alto.
        # Así cualquier pérdida real calculada será menor y podrá reemplazarlo.
        best_pinn = float('inf')

        # En ML y deep learning, una época (epochs) significa un recorrido completo por todos los datos de entrenamiento.
        # En la práctica no se suelen pasar todas las filas de golpe, sería poco eficiente. Lo que se hace es dividir
        # los datos en lotes más pequeños llamados batches.
        # Así, dentro de una 'época', el modelo procesa lote por lote, calcula el error, ajusta un poco los pesos, y pasa al siguiente lote, hasta completar así todo el dataset.
        # De esta manera, cuando termina de ver todas las filas una vez se dice que ha completado una época

        # Una vez sé esto, entro en el bucle de entrenamiento por epochs (épocas), 
        # que son las repeticiones completas sobre los datos de entrenamiento.
        # Éste es un valor arbitrario, yo decido cuántas veces quiero que el modelo repase los datos.
        # Los valores de referencia suelen estar entre 100-200, he escogido 150.

        for epoch in range(epochs): # Desde 1 hasta 150 bucle
            self.models['PINN'].train() # Activa el modo "entrenamiento" de la red PINN.
            opt_pinn.zero_grad() # El optimizador (AdamW en este caso) guarda los gradientes de cada iteración.
            # (we typically want to explicitly set the gradients to zero before starting to do backpropagation) *está epxlicado en el blog de machine learning*
            pred = self.models['PINN'](X_tr_t)# Forward pass: el modelo recibe los datos de entrenamiento (X_tr_t)
            # y genera predicciones de biomasa. Aquí todavía no hay ajuste, solo predice con los pesos actuales.
            loss = self.models['PINN'].bio_loss(pred, y_tr_t) # Calcula la función de pérdida, en este caso la "bio_loss",
            # que combina el error cuadrático medio (MSE) con una penalización biológica (no permitir biomasa negativa).
            # Cuanto mayor sea la diferencia entre predicción y valores reales, mayor será la pérdida.
            loss.backward() # Backpropagation: a partir de la pérdida, PyTorch calcula los gradientes
            # (derivadas parciales) de cada peso de la red respecto a ese error.
            # Esto es lo que le permite saber cómo ajustar los pesos.
            torch.nn.utils.clip_grad_norm_(self.models['PINN'].parameters(), 1.0) # Gradient clipping: limita el tamaño máximo de los gradientes a 1.0.
            # Evita el problema de "exploding gradients", donde los valores se vuelven enormes y desestabilizan el entrenamiento.
            opt_pinn.step() # Paso del optimizador: AdamW utiliza los gradientes calculados en .backward()
            # para actualizar los pesos de la red y reducir la pérdida.
            # Este es el momento real en que la red "aprende" ajustando sus parámetros.
            
            # -----------------------------------------------
            # Early selection
            # -----------------------------------------------

            # Se trata de una técnica de regularización para evitar el sobreajuste.
            # Cada 50 épocas hago una pausa para evaluar el modelo en el conjunto de validación.
            # El objetivo no es entrenar (no se actualizan pesos), sino comprobar si el modelo
            # realmente está mejorando en datos NO vistos durante el entrenamiento.
            # Al finalizar todas las épocas, selecciono el mejor punto y no me quedo sólo con los pesos de la última época.
     
            if epoch % 50 == 0: # cada 50 epochs
                self.models['PINN'].eval() # Modo evaluación: desactiva Dropout y fija BatchNorm (usa estadísticas acumuladas).
              # Así mido el rendimiento "real" sin ruido de regularización.

                with torch.no_grad():
                    # Bloque sin gradientes: más rápido y ahorra memoria (no es necesario entrenar aquí).
                    val_pred = self.models['PINN'](X_val_t)
                    val_loss = nn.MSELoss()(val_pred.squeeze(), y_val_t) # Métrica de validación: MSE sobre el conjunto de validación interno (20% de train).
                    # Esta pérdida no ajusta pesos, solo la usamos para monitorizar y seleccionar el mejor modelo.

                    if val_loss < best_pinn:
                        # Early model selection: si mejora la pérdida de validación,
                        # guardo una copia de los pesos actuales como "mejores hasta ahora".
                        best_pinn = val_loss
                        best_state = self.models['PINN'].state_dict().copy()
                     
                print(f"   PINN Epoch {epoch}: {val_loss.item():.4f}")
        
        self.models['PINN'].load_state_dict(best_state)
         # la idea de esto es quedarme con los mejores pesos de la red despues de pasar por todas las epochs.
         # Eso significa que el modelo final no son los pesos de la última época, sino los pesos de la época que mejor funcionó en validación
        
        # -----------------------------------------LSTM------------------------------------------------------------------------------------------------------------------------------------
        
        self.models['LSTM'] = CompactLSTM(X_entrenamiento.shape[1])
        
        # Creao el modelo LSTM indicando cuántas features hay por paso temporal.

        opt_lstm = torch.optim.AdamW(self.models['LSTM'].parameters(), lr=0.001, weight_decay=0.01)
        
        # Optimizador AdamW también para el LSTM (buen rendimiento)
        # HIPERPARÁMETROS SELECCIONADOS:
        # - lr=0.001: Learning rate (tasa de aprendizaje)
        #   * Valor conservador que balancea velocidad de convergencia vs. estabilidad
        #   * Para LSTMs, valores típicos están en el rango [0.0001, 0.01]
        # - weight_decay=0.01: Coeficiente de regularización L2
        #   * Penaliza pesos grandes para prevenir overfitting
        
        X_tr_seq, X_val_seq = X_tr_t.unsqueeze(1), X_val_t.unsqueeze(1)
        # Las redes LSTM esperan tensores con dimensionalidad específica:
        # INPUT SHAPE REQUERIDO: (batch_size, sequence_length, input_features)
        # TRANSFORMACIÓN APLICADA:
        # - Datos originales: (N_samples, N_features) → Tensor 2D
        # - Después de unsqueeze(1): (N_samples, 1, N_features) → Tensor 3D
        # - unsqueeze(1) añade una dimensión en la posición 1 (sequence_length = 1)
        # - Esto simula secuencias de longitud unitaria para cada muestra
        best_lstm = float('inf') # variable que inicializo con infinito para que cualquier loss inicial sea considerado como mejora.

        # Hago lo mismo que en la red neuronal anterior:
        # bucle de entrenamiento basado en epochs
        for epoch in range(epochs):
            self.models['LSTM'].train() #Activa el modo de entrenamiento del modelo
            opt_lstm.zero_grad() # PyTorch acumula gradientes por defecto en .grad attributes
            # Necesario limpiar gradientes de la iteración anterior
            pred = self.models['LSTM'](X_tr_seq).squeeze()# Forward pass: el modelo recibe los datos de entrenamiento (X_tr_t)
            # y genera predicciones de biomasa. Aquí todavía no hay ajuste, solo predice con los pesos actuales.
            loss = nn.MSELoss()(pred, y_tr_t) # Función de pérdida: Error Cuadrático Medio entre predicción y objetivo.
            # # En la LSTM utilizo MSE "puro" (no la bio_loss), porque aquí no hay que imponer la restricción de no-negatividad
            # (esa restricción está sólo  en la PINN).

            # Porque MSE y no otro parametro como MAE o Huber Loss?
            # porque me da la gana la verdad, además:

            #  PROPIEDADES DEL MSE:
            # - Diferenciable en todo punto 
            # - Penaliza cuadráticamente los errores grandes
            # - Unidades: cuadrado de las unidades de la variable objetivo
            # - Sensible a outliers debido a la penalización cuadrática

            loss.backward() # Backpropagation: calcula los gradientes de la pérdida con respecto a TODOS los parámetros entrenables de la LSTM.
            # EXPLICACIÓN TEÓRICA:
            # PROBLEMA: Exploding Gradients en RNNs
            # - Los gradientes pueden crecer exponencialmente durante BPTT
            # - Esto causa inestabilidad numérica y divergencia del entrenamiento
            # - Especialmente problemático en secuencias largas
            # La solución? -> Clipping de Gradiente

            torch.nn.utils.clip_grad_norm_(self.models['LSTM'].parameters(), 1.0) # Clipping de gradiente (norma L2) a 1.0 para estabilizar el entrenamiento.
            # Evita "exploding gradients" típicos en RNN/LSTM cuando hay dependencia temporal larga.

            opt_lstm.step()
             # Paso de optimización: AdamW actualiza los pesos usando los gradientes calculados.


            # -----------------------------------------------
            # Early selection
            # -----------------------------------------------

            # Se trata de una técnica de regularización para evitar el sobreajuste.
            # Cada 50 épocas hago una pausa para evaluar el modelo en el conjunto de validación.
            # El objetivo no es entrenar (no se actualizan pesos), sino comprobar si el modelo
            # realmente está mejorando en datos NO vistos durante el entrenamiento.
            # Al finalizar todas las épocas, selecciono el mejor punto y no me quedo sólo con los pesos de la última época.
     

            if epoch % 50 == 0:
                 # Cada 50 épocas hago una evaluación rápida en el conjunto de validación para monitorizar overfitting
                 # y guardar el mejor estado de los pesos (early selection).
                 
                self.models['LSTM'].eval()
                 # Modo evaluación: desactiva dropout (todas las neuronas activas) y fija batchnorm en modo evaluación (Usa estadísticas poblacionales (media/var globales)).
    
                with torch.no_grad():
                    # Bloque sin gradientes: así ahorro memoria y tiempo en validación
                    val_pred = self.models['LSTM'](X_val_seq).squeeze()  # Forward en validación: misma forma (batch_val,)
                    val_loss = nn.MSELoss()(val_pred, y_val_t) 
                     # Pérdida de validación con MSE: métrica consistente con el entrenamiento.
                    
                    if val_loss < best_lstm:
                        # Si mejora la mejor pérdida vista, guardo los pesos actuales como "los mejores hasta ahora"
                        best_lstm = val_loss
                        best_state_lstm = self.models['LSTM'].state_dict().copy()
                        # state_dict() = diccionario con todos los tensores de pesos. Hago una copia para congelarlos.

                print(f"   LSTM Epoch {epoch}: {val_loss.item():.4f}")
        
        self.models['LSTM'].load_state_dict(best_state_lstm)
     # Al terminar todas las épocas, restauro el MEJOR estado de la LSTM (no el último), lo que mitiga el riesgo de sobreajuste tardío.



        # ------------------------------------------------------------
        #  5) CÁLCULO DE ERRORES DE VALIDACIÓN Y ASIGNACIÓN DE PESOS
        # ------------------------------------------------------------
        # Una vez entrenados todos los modelos, necesito saber qué tan bien lo hace cada uno
        # en el conjunto de validación (datos no vistos durante el entrenamiento).
        # Para eso, calculo el error cuadrático medio (MSE) de cada modelo.

        for name, model in self.models.items():
            # Recorro todos los modelos entrenados: Lineal, Ridge, RandomForest, XGBoost, PINN y LSTM.
            if name in ['PINN', 'LSTM']:
                # Las redes neuronales (PINN y LSTM) están en PyTorch y no usan .predict()
                # como los modelos de Scikit-learn, así que las trato de forma especial.
                model.eval()
                 # Paso a modo evaluación:
                 #   - Dropout desactivado
                 #   - BatchNorm fija estadísticas globales
                 # Esto asegura medir el rendimiento real.
                with torch.no_grad():
                     # Desactivo el cálculo de gradientes para ahorrar memoria y tiempo,
                     # ya que no necesito entrenar en esta fase, solo evaluar.
                    if name == 'PINN':
                        # La PINN recibe un tensor con forma (batch, features).
                        pred = model(X_val_t).squeeze().numpy()
                    else:
                        # La LSTM requiere secuencias, por eso uso X_val_seq (con dimensión extra)
                        pred = model(X_val_seq).squeeze().numpy()
            else:
                 # Para los modelos clásicos (Lineal, Ridge, RF, XGBoost),
                 # sí puedo usar directamente el método .predict() de Scikit-learn/XGBoost.
                pred = model.predict(X_val)
            
            self.val_scores[name] = mean_squared_error(y_val, pred)
            # Guardo el error cuadrático medio (MSE) en un diccionario con el nombre del modelo como clave.
            # Esto me dará una medida objetiva de cuál modelo predice mejor en validación.
        
        # Al terminar el bucle, tengo un diccionario con los MSE de todos los modelos.
        print(f"✅ Validación completada! Val MSE: {self.val_scores}")
        # Con estos errores después calcularé los pesos del ensemble
        
        
        
 # ------------------------------------------------------------------
# 6) CÁLCULO DE PESOS DEL ENSEMBLE (en función del error de validación)
# ------------------------------------------------------------------
# (ya lo expliqué arriba)
# Paso 1: transformo cada MSE en "calidad" usando el inverso 1/MSE. 
#         Si un modelo tiene MSE pequeño → 1/MSE grande → más peso.
# Paso 2: normalizo esas calidades para que todos los pesos sumen 1 (distribución de probabilidad).
#         Así evito escalas arbitrarias y mantengo interpretabilidad.

        inverse_errors = {k: 1/v for k, v in self.val_scores.items() if v > 0}
        # Diccionario: modelo -> 1/MSE. (Descarto MSE <= 0 por seguridad numérica).
        total = sum(inverse_errors.values())
        # Suma total de "calidades" (1/MSE) para normalizar.
        self.weights = {k: v/total for k, v in inverse_errors.items()}
        # Normalización: peso_k = (1/MSE_k) / Σ_j (1/MSE_j).
# ------------------------------------------------------------------
# 7) PREDICCIÓN CON EL ENSEMBLE
# ------------------------------------------------------------------
# Procedimiento:    
# Obtengo la predicción de CADA modelo sobre X_validacion y al final, combino todas las predicciones con los pesos calculados arriba.
#
#  ensemble = Σ_k (peso_k * pred_k)
#
# detalle: Redes (PyTorch) requieren tensores y modo eval(); modelos sklearn usan .predict()
    def predict(self, X_validacion):
        predictions = {} ## Guardará predicciones por modelo: {'Linear': y_hat_lin, 'XGBoost': y_hat_xgb, ...}
        
        for name, model in self.models.items():
            if name in ['PINN', 'LSTM']:
                # Modelos en PyTorch: debo pasar a modo evaluación y desactivar gradientes
                model.eval()
                with torch.no_grad():
                    # Convierto el DataFrame a tensor float32
                    X_tensor = torch.FloatTensor(X_validacion.values)
                    if name == 'PINN':
                        # La PINN espera tensores 2D: (batch, features)
                        pred = model(X_tensor).squeeze().numpy()
                    else:
                        # La LSTM espera tensores 3D: (batch, seq_len, features)
                        # Aquí uso seq_len=1 (ventana temporal de 1 paso)
                        pred = model(X_tensor.unsqueeze(1)).squeeze().numpy()
            else:
                # Modelos clásicos (scikit-learn / XGBoost) usan .predict() directamente sobre DataFrame
                pred = model.predict(X_validacion)
            
            predictions[name] = pred # Guardo la predicción de cada modelo

        
        # ---------------------------
        # Combinación ponderada (Ensemble)
        # ---------------------------
        # Creo un vector de ceros y voy sumando cada predicción multiplicada por su peso.
        ensemble = np.zeros(len(X_validacion))
        for name, weight in self.weights.items():
            # Sumatorio: y_ensemble = Σ_k (peso_k * y_pred_k)
            ensemble += weight * predictions[name]
            # Sumatorio: y_ensemble = Σ_k (peso_k * y_pred_k)
        predictions['Ensemble'] = ensemble
        
        return predictions

# ========================================================================
# EVALUACIÓN INTEGRAL Y ANÁLISIS DE RESULTADOS
# ========================================================================

def evaluate_models(y_true, predictions, title="Evaluation"):
    print(f"\n📊 {title.upper()}")
    print("="*50)
    

    # MÉTRICAS IMPLEMENTADAS:
    
    # 1. R² (Coeficiente de Determinación):
    #   - Rango: (-∞, 1], donde 1 = predicción perfecta
    #   - Interpretación: Proporción de varianza explicada por el modelo
    #   - Fórmula: R² = 1 - (SS_res / SS_tot)
    #   - Ventaja: Normalizado, fácil interpretación
    #   - Limitación: Puede ser engañoso con datos no lineales
    
    # 2. RMSE (Root Mean Square Error):
    #   - Unidades: Mismas que la variable objetivo
    #   - Sensible a outliers (penalización cuadrática)
    #   - Útil para comparar modelos en el mismo dataset
    #   - Interpretación directa en términos físicos del problema
    
    # 3. MAE (Mean Absolute Error):
    #   - Menos sensible a outliers que RMSE
    #   - Métrica robusta para evaluación general
    #   - Interpretación: Error promedio en valor absoluto
    
    # 4. MAPE (Mean Absolute Percentage Error):
    #   - Métrica relativa independiente de escala
    #   - Útil para comparar performance entre diferentes datasets
    #   - Limitación: Problemático cuando y_true ≈ 0
    
    # 5. NSE (Nash-Sutcliffe Efficiency):
    #   - Métrica específica para modelado hidrológico/ambiental
    #   - Compara modelo vs. predicción con media histórica
    #   - NSE = 1: predicción perfecta; NSE = 0: tan bueno como la media
    
    # 6. BIAS (Sesgo Relativo):
    #   - Mide tendencia sistemática del modelo
    #   - Bias > 0: sobreestimación; Bias < 0: subestimación
    #   - Crítico para aplicaciones donde la dirección del error importa
    
    #  DETECCIÓN DE OVERFITTING:
    #La categorización de riesgo (LOW/MEDIUM/HIGH) se basa en umbrales empíricos:
    #- R² > 0.99: ALTO riesgo (posible memorización)
    #- R² > 0.97: MEDIO riesgo (requiere validación adicional)
    #- R² ≤ 0.97: BAJO riesgo (generalización aceptable)

    results = {}
    for name, pred in predictions.items():
        r2 = r2_score(y_true, pred)
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        mae = mean_absolute_error(y_true, pred)
        mape = np.mean(np.abs((y_true - pred) / (y_true + 1e-10))) * 100
        
        # NSE y bias
        nse = 1 - (np.sum((y_true - pred)**2) / np.sum((y_true - np.mean(y_true))**2))
        bias = (np.sum(pred - y_true) / np.sum(y_true)) * 100
        
        # Overfitting 
        risk = "HIGH" if r2 > 0.99 else "MEDIUM" if r2 > 0.97 else "LOW"
        
        results[name] = {
            'R²': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape,
            'NSE': nse, 'Bias': bias, 'Risk': risk
        }
        
        print(f"🔸 {name.upper()}")
        print(f"   R²={r2:.4f}, RMSE={rmse:.4f}, MAPE={mape:.1f}%")
        print(f"   NSE={nse:.4f}, Bias={bias:.2f}%, Risk={risk}")
    
    return results

# GRÁFICOS IMPLEMENTADOS:
    
#    1. SCATTER PLOT (Predicho vs. Observado):
#       - Evalúa la correlación lineal entre predicciones y valores reales
#       - Línea diagonal roja: predicción perfecta (y = x)
#       - Desviaciones de la diagonal indican sesgo sistemático
#       - Dispersión alrededor de la línea indica precisión del modelo
    
#    2. BAR PLOT (Comparación de R²):
#       - Codificación por colores para identificación rápida de riesgo:
#         * Verde: R² ≤ 0.97 (generalización aceptable)
#         * Naranja: 0.97 < R² ≤ 0.99 (posible overfitting leve)
#         * Rojo: R² > 0.99 (riesgo alto de overfitting)
    
#    3. RMSE COMPARISON:
#       - Métrica en unidades originales para interpretación práctica
#       - Permite evaluar significancia práctica vs. estadística
    
#    4. ANÁLISIS DE RESIDUOS:
#       - Gráfico fundamental para validación de modelos de regresión
#       - Patrones en residuos indican violación de supuestos:
#         * Heteroscedasticidad: varianza no constante
#         * No linealidad: relaciones no capturadas
#         * Autocorrelación: dependencias temporales no modeladas
    
#    5. DISTRIBUCIÓN DE RESIDUOS:
#       - Test visual de normalidad (supuesto para intervalos de confianza)
#       - Ajuste de curva normal para comparación cuantitativa
#       - Desviaciones de normalidad pueden indicar problemas del modelo
    
#    6. COMPARACIÓN MULTI-MÉTRICA:
#       - Visualización lado a lado de R² y NSE
#       - Permite evaluación holística considerando múltiples criterios
#       - Identifica modelos con rendimiento balanceado vs. especializados

def create_plots(y_true, predictions, results):
    """Create essential publication plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Best model predictions
    best_model = max(results.keys(), key=lambda k: results[k]['R²'])
    pred_best = predictions[best_model]
    
    axes[0,0].scatter(y_true, pred_best, alpha=0.6, s=20)
    min_val, max_val = min(y_true.min(), pred_best.min()), max(y_true.max(), pred_best.max())
    axes[0,0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0,0].set_xlabel('Observed (g/L)')
    axes[0,0].set_ylabel('Predicted (g/L)')
    axes[0,0].set_title(f'{best_model} - R²={results[best_model]["R²"]:.3f}')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: R² comparison
    models = list(results.keys())
    r2_vals = [results[m]['R²'] for m in models]
    bars = axes[0,1].bar(models, r2_vals)
    for bar, r2 in zip(bars, r2_vals):
        bar.set_color('red' if r2 > 0.99 else 'orange' if r2 > 0.97 else 'green')
    axes[0,1].set_ylabel('R² Score')
    axes[0,1].set_title('Model Comparison')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: RMSE comparison
    rmse_vals = [results[m]['RMSE'] for m in models]
    axes[0,2].bar(models, rmse_vals)
    axes[0,2].set_ylabel('RMSE (g/L)')
    axes[0,2].set_title('RMSE Comparison')
    axes[0,2].tick_params(axis='x', rotation=45)
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: Residuals
    residuals = y_true - pred_best
    axes[1,0].scatter(pred_best, residuals, alpha=0.6, s=20)
    axes[1,0].axhline(0, color='red', linestyle='--')
    axes[1,0].set_xlabel('Predictions')
    axes[1,0].set_ylabel('Residuals')
    axes[1,0].set_title('Residual Analysis')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 5: Residual distribution
    axes[1,1].hist(residuals, bins=30, density=True, alpha=0.7)
    mu, sigma = stats.norm.fit(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1,1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2)
    axes[1,1].set_xlabel('Residuals')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Residual Distribution')
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Multi-metric radar (simplified)
    metrics = ['R²', 'NSE']
    model_names = list(results.keys())[:4]  # Top 4 models
    
    x = np.arange(len(model_names))
    width = 0.35
    
    r2_vals = [results[m]['R²'] for m in model_names]
    nse_vals = [results[m]['NSE'] for m in model_names]
    
    axes[1,2].bar(x - width/2, r2_vals, width, label='R²', alpha=0.8)
    axes[1,2].bar(x + width/2, nse_vals, width, label='NSE', alpha=0.8)
    axes[1,2].set_xlabel('Models')
    axes[1,2].set_ylabel('Score')
    axes[1,2].set_title('Multi-Metric Comparison')
    axes[1,2].set_xticks(x)
    axes[1,2].set_xticklabels(model_names, rotation=45)
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

#  ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
    
#    1. INTERPRETABILIDAD DEL MODELO:
#       - Identificar qué variables son más relevantes para las predicciones
#       - Facilitar la comprensión del modelo por parte de expertos del dominio
#       - Cumplir con requisitos de explicabilidad en aplicaciones críticas
    
#    2. REDUCCIÓN DIMENSIONAL:
#       - Identificar características redundantes o irrelevantes
#       - Optimizar el modelo eliminando features de baja importancia
#       - Reducir complejidad computacional y riesgo de overfitting
    
#    3. VALIDACIÓN CIENTÍFICA:
#       - Verificar que el modelo identifica relaciones conocidas del dominio
#       - Descubrir nuevas relaciones potencialmente importantes
#       - Contrastar resultados con conocimiento experto previo
    
#    4. FEATURE ENGINEERING:
#       - Guiar la creación de nuevas características derivadas
#       - Informar decisiones sobre transformaciones de variables
#       - Identificar interacciones importantes entre variables
    
#    MÉTODOS DE IMPORTANCIA:
#    - Tree-based models: Importancia basada en reducción de impureza (Gini, entropy)
#    - Linear models: Coeficientes normalizados o métodos de permutación
#    - Neural networks: Gradientes, saliency maps, SHAP values
def analyze_importance(model, feature_names):
    """Feature importance analysis"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔍 TOP 10 FEATURES:")
        for _, row in importance_df.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    return None

# ========================================================================
# MAIN PIPELINE
# ========================================================================

def run_compact_research():
    
    # 1. CARGA Y LIMPIA LOS DATOS
    manager = SmartDataManager()
    data, leakage = manager.carga_limpia_datos()
    # Se cargan los datos originales y se hace una limpieza inicial.
    # También se identifican y eliminan variables con posible "data leakage"
    # (features que contienen información del futuro o de la variable objetivo
    # y que falsearían el entrenamiento si se incluyeran).
    
    # 2. FEATURE ENGINEERING
    engine = BioFeatureEngine()
    df_features = engine.create_features(data)
    # Se construyen nuevas variables derivadas (ej. interacciones, normalizaciones,
    # indicadores biológicos) a partir de los datos originales.
    # El objetivo es enriquecer el dataset con información más expresiva
    # para mejorar la capacidad predictiva de los modelos.

    # 3. SEPARAR LOS DATOS
    X_entrenamiento_s, X_validacion_s, Y_entrenamiento_s, Y_validacion_s, Y_entrenamiento_orig, Y_validacion_orig = engine.seleccion_y_preparacion_features(df_features)
    # Se dividen los datos en conjunto de entrenamiento (para ajustar los modelos)
    # y conjunto de validación (para evaluar su desempeño en datos no vistos).
    # "_s" indica que las variables han sido escaladas (normalizadas).
    # También se guardan los valores originales (sin escalar) de Y para poder
    # interpretar los resultados en unidades reales más adelante.

    # 4. ENTRENAR LOS MODELOS
    system = CompactMultiModel()
    system.train_all(X_entrenamiento_s, Y_entrenamiento_s)
    # Se inicializa el sistema que contiene todos los modelos (lineales, árboles,
    # XGBoost, PINN, LSTM, etc.) y se entrena cada uno sobre los datos escalados.
    # Cada modelo aprende relaciones distintas, y después se combinarán en un ensemble.

    # 5. OBTENER PREDICCIONES
    predictions_scaled = system.predict(X_validacion_s)
    # Se obtienen predicciones de todos los modelos, pero todavía en la escala
    # normalizada usada durante el entrenamiento (ej. entre 0 y 1).

    
    # 6. DESNORMALIZACIÓN (INTERPRETABILIDAD)
    # Se transforman las predicciones a la escala original (g/mol o g/L),
    # para que tengan un significado físico y se puedan comparar con las observaciones reales.
    predictions_orig = {}
    for name, pred_scaled in predictions_scaled.items():
        pred_orig = engine.scalers['target'].inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        # con esto obtengo la predicción final de nuevo en g_mol para poder interpretarla de manera correcta.
        predictions_orig[name] = pred_orig
    
    # 7. EVALUACIÓN FINAL DE MODELOS
    results = evaluate_models(Y_validacion_orig, predictions_orig, "COMPACT SYSTEM RESULTS")
    # Se evalúan todos los modelos usando varias métricas (R², RMSE, MAE, MAPE, NSE, Bias).
    # Esto permite un diagnóstico completo: ajuste general, error absoluto, error relativo,
    # sesgos sistemáticos y riesgo de sobreajuste.
    
    # 8. VISUALIZACIÓN
    create_plots(Y_validacion_orig, predictions_orig, results)
    # Se generan gráficos de apoyo para comunicar resultados:
    #   - Diagrama de paridad observado vs predicho.
    #   - Comparaciones de R² y RMSE entre modelos.
    #   - Análisis de residuos y distribución.
    #   - Comparativa multi-métrica.
    # Estos gráficos son más comunicativos que las métricas numéricas aisladas.

    
    # 9. FEATURES IMPORTANTES
    if 'XGBoost' in system.models:
        importance = analyze_importance(system.models['XGBoost'], engine.selected_features)
    # Para modelos que lo permiten (ej. RandomForest, XGBoost), se analiza la
    # importancia relativa de cada variable de entrada.
    # Esto aporta interpretabilidad: ¿qué factores influyen más en la predicción de biomasa?

    
    # 10. MEJOR MODELO, R2 Y RMSE
    best_model = max(results.keys(), key=lambda k: results[k]['R²'])
    best_r2 = results[best_model]['R²']
    best_rmse = results[best_model]['RMSE']
    # Se selecciona automáticamente el modelo con mejor R² en validación.
    # También se guardan sus métricas clave (R² y RMSE).

    
    print(f"\n🎉 MODELO PREDICTIVO COMPLETADO!")
    print("="*40)
    print(f"🏆 Mejor Modelo: {best_model}")
    print(f"📊 Performance: R²={best_r2:.4f}, RMSE={best_rmse:.4f}")
    print(f"🛡️ Anti-overfitting: {results[best_model]['Risk']} risk")
    print(f"📈 Leakage eliminado: {len(leakage)} features")
    print(f"🔬 Features utilizadas: {len(engine.selected_features)}")
    
    return {
        'system': system, 'results': results, 'engine': engine,
        'best_model': best_model, 'leakage_removed': len(leakage)
    }

# ========================================================================
# EJECUCIÓN
# ========================================================================

if __name__ == "__main__":
    
    final_results = run_compact_research()
    
    if final_results:
        print(f"\n✅ SUCCESS!")
