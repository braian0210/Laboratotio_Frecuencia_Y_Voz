# Laboratotio_Frecuencia_Y_Voz

Objetivos

Capturar y procesar señales de voz masculinas y femeninas.

 Aplicar la Transformada de Fourier como herramienta de análisis espectral de la
voz.

 Extraer parámetros característicos de la señal de voz: frecuencia fundamental,
frecuencia media, brillo, intensidad, jitter y shimmer.

 Comparar las diferencias principales entre señales de voz de hombres y mujeres
a partir de su análisis en frecuencia.

 Desarrollar conclusiones sobre el comportamiento espectral de la voz humana
en función del género. 

PARTE A – Adquisición de las señales de voz

1. Grabar con un micrófono la misma frase corta (aprox. 5 segundos) en 6
personas distintas: 3 hombres y 3 mujeres. Para esto pueden usar los
micrófonos de sus teléfonos inteligentes y configurar las características de
muestreo para que sean las mismas en todos los dispositivos.

2. Guardar cada archivo de voz en formato .wav con un nombre identificador
claro (ejemplo: mujer1.wav, hombre2.wav).

3. Importar las señales de voz en Python y graficarlas en el dominio del tiempo.
   
4. Calcular la Transformada de Fourier de cada señal y graficar su espectro de
magnitudes frecuenciales.

5. Identificar y reportar las siguientes características de cada señal:
    
a. Frecuencia fundamental.

b. Frecuencia media.

c. Brillo.

d. Intensidad (energía).

PARTE B – Medición de Jitter y Shimmer 

Seleccione una de las grabaciones realizadas en la Parte A por cada género
(una voz de hombre y una de mujer).

   Aplique un filtro pasa-banda en el rango de la voz (80–400 Hz para
hombres, 150–500 Hz para mujeres) para eliminar ruido no deseado.

Se escogió al sujeto 1 masculini y femenino para aplicar el filtro pasa banda.

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from google.colab import drive
import IPython.display as ipd
import os
drive.mount('/content/drive')

# Función para aplicar filtro pasa banda Butterworth
def filtro_pasabanda(audio, fs, lowcut, highcut, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Check if audio length is sufficient for the default order
    min_length_required = 3 * (order + 1) # Approximate minimum length for filtfilt
    if len(audio) <= min_length_required:
        print(f"Warning: Audio length ({len(audio)}) is too short for filter order {order}. Reducing order.")
        # Reduce order to the maximum possible
        new_order = max(1, (len(audio) // 3) - 1)
        print(f"Using filter order: {new_order}")
        order = new_order

    if order <= 0:
        print("Error: Audio is too short for any valid filter order.")
        return np.zeros_like(audio) # Return a zero array or handle as an error case

    b, a = butter(order, [low, high], btype='band')
    audio_filtrado = filtfilt(b, a, audio)
    return audio_filtrado

# Verificar y corregir rutas de archivos
print("=== VERIFICACIÓN DE ARCHIVOS ===")

# Rutas 
ruta_base = '/content/drive/Shareddrives/Labs procesamiento de señales/Lab 3/'
ruta_audio_fem = os.path.join(ruta_base, 'sujeto 1 fem.wav')
ruta_audio_masc = os.path.join(ruta_base, 'sujeto masc 1 1.wav')

print(f"Ruta femenina: {ruta_audio_fem}")
print(f"Ruta masculina: {ruta_audio_masc}")

# Verificar si los archivos existen
archivo_fem_existe = os.path.exists(ruta_audio_fem)
archivo_masc_existe = os.path.exists(ruta_audio_masc)

print(f"¿Archivo femenino existe? {archivo_fem_existe}")
print(f"¿Archivo masculino existe? {archivo_masc_existe}")

# Si los archivos no existen, buscar alternativas
if not archivo_fem_existe or not archivo_masc_existe:
    print("\nBuscando archivos WAV en el directorio...")
    if os.path.exists(ruta_base):
        archivos = os.listdir(ruta_base)
        archivos_wav = [f for f in archivos if f.lower().endswith('.wav')]
        print("Archivos WAV encontrados:", archivos_wav)
        
        # Si encontramos archivos WAV, usar los primeros disponibles
        if archivos_wav:
            if not archivo_fem_existe and len(archivos_wav) > 0:
                ruta_audio_fem = os.path.join(ruta_base, archivos_wav[0])
                print(f"Usando para femenino: {ruta_audio_fem}")
            if not archivo_masc_existe and len(archivos_wav) > 1:
                ruta_audio_masc = os.path.join(ruta_base, archivos_wav[1])
                print(f"Usando para masculino: {ruta_audio_masc}")
            elif not archivo_masc_existe and len(archivos_wav) > 0:
                ruta_audio_masc = os.path.join(ruta_base, archivos_wav[0])
                print(f"Usando mismo archivo para masculino: {ruta_audio_masc}")

# Variables para almacenar los audios
audio_fem = np.array([])
audio_fem_filtrado = np.array([])
audio_masc = np.array([])
audio_masc_filtrado = np.array([])
fs_fem = 0
fs_masc = 0

# Cargar y filtrar audio femenino
print("\n=== PROCESANDO AUDIO FEMENINO ===")
try:
    if os.path.exists(ruta_audio_fem):
        fs_fem, audio_fem = wavfile.read(ruta_audio_fem)
        print(f"Audio femenino cargado: {len(audio_fem)} muestras, frecuencia: {fs_fem} Hz")
        
        # Convertir a float32 y normalizar si es necesario
        audio_fem = audio_fem.astype(np.float32)
        
        # Si el audio es stereo, tomar solo un canal
        if len(audio_fem.shape) > 1:
            audio_fem = audio_fem[:, 0]
            print("Audio stereo detectado, usando solo el primer canal")
        
        # Aplicar filtro
        if len(audio_fem) > 0:
            print("Aplicando filtro pasa banda (150-500 Hz)...")
            audio_fem_filtrado = filtro_pasabanda(audio_fem, fs_fem, 150, 500)
            print("Filtro aplicado exitosamente")
        else:
            print("Error: Audio femenino está vacío")
    else:
        print(f"Error: Archivo no encontrado - {ruta_audio_fem}")
        
except Exception as e:
    print(f"Error cargando o filtrando audio femenino: {e}")

# Cargar y filtrar audio masculino
print("\n=== PROCESANDO AUDIO MASCULINO ===")
try:
    if os.path.exists(ruta_audio_masc):
        fs_masc, audio_masc = wavfile.read(ruta_audio_masc)
        print(f"Audio masculino cargado: {len(audio_masc)} muestras, frecuencia: {fs_masc} Hz")
        
        # Convertir a float32 y normalizar si es necesario
        audio_masc = audio_masc.astype(np.float32)
        
        # Si el audio es stereo, tomar solo un canal
        if len(audio_masc.shape) > 1:
            audio_masc = audio_masc[:, 0]
            print("Audio stereo detectado, usando solo el primer canal")
        
        # Aplicar filtro
        if len(audio_masc) > 0:
            print("Aplicando filtro pasa banda (80-400 Hz)...")
            audio_masc_filtrado = filtro_pasabanda(audio_masc, fs_masc, 80, 400)
            print("Filtro aplicado exitosamente")
        else:
            print("Error: Audio masculino está vacío")
    else:
        print(f"Error: Archivo no encontrado - {ruta_audio_masc}")
        
except Exception as e:
    print(f"Error cargando o filtrando audio masculino: {e}")

# Graficar señal original y filtrada
print("\n=== GENERANDO GRÁFICOS ===")
plt.figure(figsize=(14, 8))

# Gráfico para voz femenina
plt.subplot(2, 1, 1)
if len(audio_fem) > 0:
    # Mostrar solo una parte para mejor visualización (primeros 10000 puntos)
    muestras_a_mostrar = min(10000, len(audio_fem))
    tiempo = np.arange(muestras_a_mostrar) / fs_fem
    
    plt.plot(tiempo, audio_fem[:muestras_a_mostrar], color='gray', alpha=0.7, label='Original Femenina')
    
if len(audio_fem_filtrado) > 0:
    muestras_a_mostrar_filt = min(10000, len(audio_fem_filtrado))
    tiempo_filt = np.arange(muestras_a_mostrar_filt) / fs_fem
    plt.plot(tiempo_filt, audio_fem_filtrado[:muestras_a_mostrar_filt], color='magenta', linewidth=1.5, label='Filtrada (150–500 Hz)')

plt.legend()
plt.title('Filtro pasa banda Butterworth - Voz femenina')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True, alpha=0.3)

# Gráfico para voz masculina
plt.subplot(2, 1, 2)
if len(audio_masc) > 0:
    muestras_a_mostrar = min(10000, len(audio_masc))
    tiempo = np.arange(muestras_a_mostrar) / fs_masc
    
    plt.plot(tiempo, audio_masc[:muestras_a_mostrar], color='gray', alpha=0.7, label='Original Masculina')
    
if len(audio_masc_filtrado) > 0:
    muestras_a_mostrar_filt = min(10000, len(audio_masc_filtrado))
    tiempo_filt = np.arange(muestras_a_mostrar_filt) / fs_masc
    plt.plot(tiempo_filt, audio_masc_filtrado[:muestras_a_mostrar_filt], color='blue', linewidth=1.5, label='Filtrada (80–400 Hz)')

plt.legend()
plt.title('Filtro pasa banda Butterworth - Voz masculina')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Reproducir el audio filtrado
print("\n=== REPRODUCCIÓN DE AUDIO ===")
print("Voz femenina filtrada:")
if len(audio_fem_filtrado) > 0:
    print("Reproduciendo audio femenino filtrado...")
    ipd.display(ipd.Audio(audio_fem_filtrado, rate=fs_fem))
else:
    print("No hay audio filtrado para reproducir para voz femenina.")

print("\nVoz masculina filtrada:")
if len(audio_masc_filtrado) > 0:
    print("Reproduciendo audio masculino filtrado...")
    ipd.display(ipd.Audio(audio_masc_filtrado, rate=fs_masc))
else:
    print("No hay audio filtrado para reproducir para voz masculina.")

# Resumen final
print("\n=== RESUMEN FINAL ===")
print(f"Audio femenino - Original: {len(audio_fem)} muestras, Filtrado: {len(audio_fem_filtrado)} muestras")
print(f"Audio masculino - Original: {len(audio_masc)} muestras, Filtrado: {len(audio_masc_filtrado)} muestras")

```
Obteniéndose 

```
<img width="1389" height="790" alt="image" src="https://github.com/user-attachments/assets/638fdd5a-5913-4899-b4f1-e26e5c491c00" />

```


 Medición del Jitter (variación en la frecuencia fundamental):

   Detecte los periodos de vibración de la señal (usando cruces por cero
  o picos sucesivos).

   Calcule los periodos Ti de la señal de voz.

   Obtenga el jitter absoluto: 

  <img width="310" height="99" alt="image" src="https://github.com/user-attachments/assets/566fa616-b053-4c6c-88d8-afada140442b" />

  Calcule el jitter relativo (%): 

  <img width="316" height="71" alt="image" src="https://github.com/user-attachments/assets/bc6473b8-1af2-4c48-809d-e39bc3da8210" />

 Medición del Shimmer (variación en la amplitud):

   Detecte los picos de amplitud Ai en cada ciclo.
  
   Obtenga el shimmer absoluto: 

  <img width="313" height="90" alt="image" src="https://github.com/user-attachments/assets/994db285-de06-408b-827f-66651a5cb0a1" />

   Calcule el shimmer relativo (%): 
  
  <img width="339" height="78" alt="image" src="https://github.com/user-attachments/assets/bc624b39-5211-4d98-9bf9-40f5343dba37" />

 Presente los valores obtenidos de jitter y shimmer para cada una de las 6
grabaciones (3 hombres, 3 mujeres). 

PARTE C – Comparación y conclusiones 

Comparar los resultados obtenidos entre las voces masculinas y femeninas.ç

 ¿Qué diferencias se observan en la frecuencia fundamental?

 ¿Qué otras diferencias notan en términos de brillo, media o intensidad?

 Redactar conclusiones sobre el comportamiento de la voz en hombres y
mujeres a partir de los análisis realizados.

 Discuta la importancia clínica del jitter y shimmer en el análisis de la voz. 





















