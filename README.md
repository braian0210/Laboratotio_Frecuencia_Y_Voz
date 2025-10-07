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

![Imagen de WhatsApp 2025-10-06 a las 21 13 03_a1233061](https://github.com/user-attachments/assets/d44bf95d-554c-4525-867c-40fa1f75aa03)


1. Grabar con un micrófono la misma frase corta (aprox. 5 segundos) en 6
personas distintas: 3 hombres y 3 mujeres. Para esto pueden usar los
micrófonos de sus teléfonos inteligentes y configurar las características de
muestreo para que sean las mismas en todos los dispositivos.



2. Guardar cada archivo de voz en formato .wav con un nombre identificador
claro (ejemplo: mujer1.wav, hombre2.wav).

<img width="190" height="185" alt="image" src="https://github.com/user-attachments/assets/db76bae1-4575-478e-8a5f-948fc4df105c" />



3. Importar las señales de voz en Python y graficarlas en el dominio del tiempo.
   

```
 import wave
import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa import effects

# === Cargar archivo de audio ===
audio = wave.open(r"/content/drive/Shareddrives/Labs procesamiento de señales/Lab 3/sujeto 1 fem.wav", "rb")

sample_frec = audio.getframerate()
n_muestras = audio.getnframes()
onda_señal = audio.readframes(-1)
num_channels = audio.getnchannels()
audio.close()

duracionaudio = n_muestras / sample_frec
signal_array = np.frombuffer(onda_señal, dtype=np.int16)
signal_array = signal_array.reshape(-1, num_channels)
times = np.linspace(0, duracionaudio, num=n_muestras, endpoint=False)

# === Señal de voz cruda ===
plt.figure(figsize=(15, 5))
plt.plot(times, signal_array[:, 0])
plt.title("Señal de voz - sujeto 1 fem (raw)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.xlim(0, duracionaudio)
plt.grid(True)
plt.show()

# === Cálculo RMS ===
ruta = r"/content/drive/Shareddrives/Labs procesamiento de señales/Lab 3/sujeto 1 fem.wav"
y, sr = librosa.load(ruta, sr=None)

frame_length = int(0.02 * sr)     # 20 ms
hop_length = int(frame_length // 4)  # solapamiento 75%

rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
rms_db = librosa.amplitude_to_db(rms, ref=1.0, amin=1e-8, top_db=80.0)
times_rms = librosa.frames_to_time(np.arange(len(rms_db)), sr=sr, hop_length=hop_length)

# === Gráfica RMS ===
plt.figure(figsize=(12,4))
plt.plot(times_rms, rms_db)
plt.title("Nivel RMS en dBFS (20 ms) - sujeto 1 fem")
plt.xlabel("Tiempo [s]")
plt.ylabel("Nivel [dBFS]")
plt.ylim(-80, 0)
plt.grid(True)
plt.show()

# === Datos generales ===
print("Sujeto1 — Frecuencia de muestreo:", sample_frec, "Hz")
print("Sujeto1 — Número de muestras:", n_muestras)
print("Sujeto1 — Duración (s):", duracionaudio)

```
   realizando este procedimiento para las 6 señales, dando como resultado la  señal directamente graficada en funcion del tiempo y luego su version en dB usando RMS.

<img width="1258" height="470" alt="image" src="https://github.com/user-attachments/assets/19cfda9b-3a64-4820-9cfb-3552c932d94b" />
<img width="1008" height="393" alt="image" src="https://github.com/user-attachments/assets/de2e3c7d-4b21-4c8e-9aba-44d58dcd244f" />
sujeto2
<img width="1258" height="470" alt="image" src="https://github.com/user-attachments/assets/da749f93-5089-4c07-bbfe-000adb153807" />
<img width="1008" height="393" alt="image" src="https://github.com/user-attachments/assets/3000ee7f-2349-47ff-aae2-da68eb0a009c" />
sujeto3
<img width="1267" height="470" alt="image" src="https://github.com/user-attachments/assets/ae303755-0525-40e9-b4b4-d0ab81ca8814" />
<img width="1008" height="393" alt="image" src="https://github.com/user-attachments/assets/e98d1324-01a5-45d0-8fa4-7bc6d47fe88b" />
sujeto4
<img width="1258" height="470" alt="image" src="https://github.com/user-attachments/assets/9578b6db-0fb5-475e-8af4-a7fec3c71b32" />
<img width="1008" height="393" alt="image" src="https://github.com/user-attachments/assets/43b511b7-2655-4602-b7b2-8811c1ad6458" />
sujeto5
<img width="1258" height="470" alt="image" src="https://github.com/user-attachments/assets/5edca332-d2a5-4c76-9f93-93e4d2ee9d76" />
<img width="1008" height="393" alt="image" src="https://github.com/user-attachments/assets/e7a636b2-6c38-47c8-8a9c-1556c0b44f1e" />
sujeto6
<img width="1267" height="470" alt="image" src="https://github.com/user-attachments/assets/0ad061b8-1884-42c1-b1f1-fe8433bbd882" />
<img width="1008" height="393" alt="image" src="https://github.com/user-attachments/assets/1d34e3a3-707c-4273-bbcb-5e3011e07697" />


   
6. Calcular la Transformada de Fourier de cada señal y graficar su espectro de magnitudes frecuenciales.
7.  Identificar y reportar las siguientes características de cada señal:
    
a. Frecuencia fundamental.

b. Frecuencia media.

c. Brillo.

d. Intensidad (energía).
```
eps = 1e-12  
signal_array = np.frombuffer(onda_señal, dtype=np.int16).reshape(-1, num_channels)
if num_channels > 1:
    signal_array = signal_array[:, 0]

# FFT (solo mitad positiva)
fft_vals = np.fft.rfft(signal_array)
fft_freq = np.fft.rfftfreq(len(signal_array), 1.0 / sample_frec)
fft_data = np.abs(fft_vals)

# Plot (semilog-x, magnitud lineal)
plt.figure(figsize=(15,4))
plt.semilogx(fft_freq, fft_data)
plt.title("Espectro de frecuencia - sujeto 1 fem")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid(True)
plt.show()

# Características
idx_f0 = np.argmax(fft_data[1:]) + 1
frecuencia_fundamental = fft_freq[idx_f0]
frecuencia_media = np.sum(fft_freq * fft_data) / np.sum(fft_data)
mask_brillo = fft_freq > 750
brillo = np.sum(fft_freq[mask_brillo] * fft_data[mask_brillo]) / np.sum(fft_data[mask_brillo])
energia = np.sum(signal_array.astype(float)**2)

print("Sujeto1 - Frecuencia fundamental:", frecuencia_fundamental, "Hz")
print("Sujeto1 - Frecuencia media:", frecuencia_media, "Hz")
print("Sujeto1 - Brillo (>750 Hz):", brillo, "Hz")
print("Sujeto1 - Energía total (tiempo):", energia)
```
donde obtuvimos las graficas en escala semilogaritmicas y los valores de cada una 

<img width="1235" height="465" alt="image" src="https://github.com/user-attachments/assets/9b95118e-ceca-4ed4-a2fc-9850f382c374" />
sujeto2
<img width="1234" height="468" alt="image" src="https://github.com/user-attachments/assets/479f1d2f-836f-4228-803c-496c1ab271dc" />
sujeto3
<img width="1234" height="465" alt="image" src="https://github.com/user-attachments/assets/90294b19-bf54-402a-97ef-eeb2f8d26da0" />
sujeto4
<img width="1232" height="466" alt="image" src="https://github.com/user-attachments/assets/6498f43a-06cd-4118-9c99-06d41966afa0" />
sujeto5
<img width="1238" height="465" alt="image" src="https://github.com/user-attachments/assets/64e591b0-b146-44ce-8931-0d3368bf5586" />
sujeto6
<img width="1242" height="474" alt="image" src="https://github.com/user-attachments/assets/91ee1126-10cd-4934-b7c5-5247630cf1f4" />


PARTE B – Medición de Jitter y Shimmer 

![Imagen de WhatsApp 2025-10-06 a las 21 35 15_35fc6513](https://github.com/user-attachments/assets/84f916cc-a7dc-470a-9066-d6965cc2d8a2)


Seleccione una de las grabaciones realizadas en la Parte A por cada género
(una voz de hombre y una de mujer).

   Aplique un filtro pasa-banda en el rango de la voz (80–400 Hz para
hombres, 150–500 Hz para mujeres) para eliminar ruido no deseado.

Se escogió al sujeto 1 masculino y femenino para aplicar el filtro pasa banda.

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


<img width="1389" height="790" alt="image" src="https://github.com/user-attachments/assets/638fdd5a-5913-4899-b4f1-e26e5c491c00" />


 Medición del Jitter (variación en la frecuencia fundamental):

   Detecte los periodos de vibración de la señal (usando cruces por cero
  o picos sucesivos).

   Calcule los periodos Ti de la señal de voz.

   Obtenga el jitter absoluto: 

  <img width="310" height="99" alt="image" src="https://github.com/user-attachments/assets/566fa616-b053-4c6c-88d8-afada140442b" />

  Calcule el jitter relativo (%): 

  <img width="316" height="71" alt="image" src="https://github.com/user-attachments/assets/bc6473b8-1af2-4c48-809d-e39bc3da8210" />

  A continuación, se muestra el código que se utilizó para calcular todas aquellas mediciones de jitter para cada sujeto femenino y masculino.

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd
from google.colab import drive
import os
drive.mount('/content/drive')

rutas_audios = {
    'Sujeto 1 Femenino': '/content/drive/Shareddrives/Labs procesamiento de señales/Lab 3/sujeto 1 fem.wav',
    'Sujeto 2 Femenino': '/content/drive/Shareddrives/Labs procesamiento de señales/Lab 3/sujeto 2 fem.wav',
    'Sujeto 3 Femenino': '/content/drive/Shareddrives/Labs procesamiento de señales/Lab 3/sujeto 3 fem.wav',
    'Sujeto 1 Masculino': '/content/drive/Shareddrives/Labs procesamiento de señales/Lab 3/sujeto masc 1 1.wav',
    'Sujeto 2 Masculino': '/content/drive/Shareddrives/Labs procesamiento de señales/Lab 3/sujeto 2 masc.wav',
    'Sujeto 3 Masculino': '/content/drive/Shareddrives/Labs procesamiento de señales/Lab 3/sujeto 3 masc.wav'
}

# Función para aplicar filtro pasa banda 
def filtro_pasabanda(audio, fs, lowcut, highcut, order=4):
    """Aplica filtro pasa banda Butterworth"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Verificar longitud del audio
    min_length_required = 3 * (order + 1)
    if len(audio) <= min_length_required:
        new_order = max(1, (len(audio) // 3) - 1)
        order = new_order

    if order <= 0:
        return np.zeros_like(audio)

    b, a = butter(order, [low, high], btype='band')
    audio_filtrado = filtfilt(b, a, audio)
    return audio_filtrado

# Función para detectar periodos usando cruces por cero
def detectar_periodos_cruces_cero(audio, fs, umbral=0.02):
    """Detecta periodos usando cruces por cero con umbral"""
    # Aplicar umbral para eliminar ruido
    audio_umbral = audio.copy()
    audio_umbral[np.abs(audio_umbral) < umbral * np.max(np.abs(audio_umbral))] = 0
    
    # Encontrar cruces por cero positivos
    cruces_cero = []
    for i in range(1, len(audio_umbral)):
        if audio_umbral[i-1] <= 0 and audio_umbral[i] > 0:
            cruces_cero.append(i)
    
    # Calcular periodos (en muestras)
    periodos_muestras = []
    for i in range(1, len(cruces_cero)):
        periodo = cruces_cero[i] - cruces_cero[i-1]
        # Filtrar periodos muy cortos o muy largos (eliminar outliers)
        if 0.001 * fs < periodo < 0.02 * fs:  # Entre 1ms y 20ms (50-1000 Hz)
            periodos_muestras.append(periodo)
    
    # Convertir a segundos
    periodos_segundos = np.array(periodos_muestras) / fs
    return periodos_segundos, cruces_cero

# Función para detectar periodos usando picos
def detectar_periodos_picos(audio, fs, distancia_minima=None):
    """Detecta periodos usando picos sucesivos"""
    if distancia_minima is None:
        distancia_minima = int(0.005 * fs)  # 5ms mínima distancia entre picos
    
    # Encontrar picos positivos
    picos, _ = find_peaks(audio, distance=distancia_minima, height=0.1*np.max(audio))
    
    # Calcular periodos (en muestras)
    periodos_muestras = []
    for i in range(1, len(picos)):
        periodo = picos[i] - picos[i-1]
        # Filtrar periodos razonables
        if 0.001 * fs < periodo < 0.02 * fs:  # Entre 1ms y 20ms
            periodos_muestras.append(periodo)
    
    # Convertir a segundos
    periodos_segundos = np.array(periodos_muestras) / fs
    return periodos_segundos, picos

# Función para calcular jitter
def calcular_jitter(periodos):
    """Calcula jitter absoluto y relativo según las fórmulas especificadas"""
    if len(periodos) < 2:
        return 0, 0, 0, 0
    
    N = len(periodos)
    
    # Calcular jitter absoluto
    suma_diferencias = 0
    for i in range(N-1):
        suma_diferencias += abs(periodos[i] - periodos[i+1])
    
    jitter_abs = suma_diferencias / (N-1)
    
    # Calcular periodo promedio
    periodo_promedio = np.mean(periodos)
    
    # Calcular jitter relativo (%)
    jitter_rel = (jitter_abs / periodo_promedio) * 100
    
    # Jitter local (variación entre periodos consecutivos)
    jitter_local = np.mean(np.abs(np.diff(periodos)))
    
    return jitter_abs, jitter_rel, periodo_promedio, jitter_local

# Función principal para procesar un audio
def procesar_audio(ruta_audio, nombre_audio, genero='femenino'):
    """Procesa un archivo de audio y calcula sus métricas de jitter"""
    print(f"\n{'='*60}")
    print(f"PROCESANDO: {nombre_audio}")
    print(f"{'='*60}")
    
    try:
        # Cargar audio
        fs, audio = wavfile.read(ruta_audio)
        print(f"Frecuencia de muestreo: {fs} Hz")
        print(f"Duración: {len(audio)/fs:.2f} segundos")
        
        # Convertir a mono si es stereo
        if len(audio.shape) > 1:
            audio = audio[:, 0]
            print("Audio convertido a mono")
        
        # Convertir a float y normalizar
        audio = audio.astype(np.float32)
        audio = audio / np.max(np.abs(audio))
        
        # Aplicar filtro pasa banda según el género
        if genero.lower() == 'femenino':
            audio_filtrado = filtro_pasabanda(audio, fs, 150, 500)
            rango_frecuencia = "150-500 Hz"
        else:
            audio_filtrado = filtro_pasabanda(audio, fs, 80, 400)
            rango_frecuencia = "80-400 Hz"
        
        print(f"Filtro aplicado: {rango_frecuencia}")
        
        # Detectar periodos usando ambos métodos
        periodos_cruces, cruces = detectar_periodos_cruces_cero(audio_filtrado, fs)
        periodos_picos, picos = detectar_periodos_picos(audio_filtrado, fs)
        
        print(f"Periodos detectados - Cruces por cero: {len(periodos_cruces)}")
        print(f"Periodos detectados - Picos: {len(periodos_picos)}")
        
        # Usar el método que detecte más periodos (generalmente más confiable)
        if len(periodos_cruces) >= len(periodos_picos):
            periodos = periodos_cruces
            metodo = "Cruces por cero"
            puntos_deteccion = cruces
        else:
            periodos = periodos_picos
            metodo = "Picos"
            puntos_deteccion = picos
        
        print(f"Método seleccionado: {metodo}")
        
        if len(periodos) < 2:
            print("ADVERTENCIA: No se detectaron suficientes periodos para calcular jitter")
            return None, None, None, None, None
        
        # Calcular métricas de jitter
        jitter_abs, jitter_rel, periodo_promedio, jitter_local = calcular_jitter(periodos)
        f0_promedio = 1 / periodo_promedio if periodo_promedio > 0 else 0
        
        print(f"\n--- RESULTADOS JITTER ---")
        print(f"Jitter absoluto: {jitter_abs*1000:.6f} ms")
        print(f"Jitter relativo: {jitter_rel:.6f} %")
        print(f"Jitter local: {jitter_local*1000:.6f} ms")
        print(f"Periodo promedio: {periodo_promedio*1000:.6f} ms")
        print(f"Frecuencia fundamental (F0): {f0_promedio:.2f} Hz")
        print(f"Número de periodos analizados: {len(periodos)}")
        
        # Visualización
        visualizar_resultados(audio, audio_filtrado, puntos_deteccion, periodos, 
                            nombre_audio, metodo, fs)
        
        return jitter_abs, jitter_rel, periodo_promedio, f0_promedio, len(periodos)
        
    except Exception as e:
        print(f"ERROR procesando {nombre_audio}: {e}")
        return None, None, None, None, None

# Función para visualizar resultados
def visualizar_resultados(audio_original, audio_filtrado, puntos_deteccion, periodos, 
                         nombre_audio, metodo, fs):
    """Genera gráficos para visualizar los resultados"""
    # Mostrar solo una parte del audio para mejor visualización
    muestras_visualizar = min(5000, len(audio_original))
    tiempo = np.arange(muestras_visualizar) / fs
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Análisis de Jitter - {nombre_audio}\n(Método: {metodo})', fontsize=16)
    
    # Gráfico 1: Señal original vs filtrada
    axes[0, 0].plot(tiempo, audio_original[:muestras_visualizar], 
                   color='gray', alpha=0.7, label='Original')
    axes[0, 0].plot(tiempo, audio_filtrado[:muestras_visualizar], 
                   color='blue', linewidth=1, label='Filtrada')
    axes[0, 0].set_title('Señal Original vs Filtrada')
    axes[0, 0].set_xlabel('Tiempo (s)')
    axes[0, 0].set_ylabel('Amplitud')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Puntos de detección
    axes[0, 1].plot(tiempo, audio_filtrado[:muestras_visualizar], 
                   color='blue', linewidth=1, label='Señal filtrada')
    
    # Marcar puntos de detección en el rango visualizado
    puntos_en_rango = [p for p in puntos_deteccion if p < muestras_visualizar]
    tiempos_puntos = np.array(puntos_en_rango) / fs
    amplitudes_puntos = audio_filtrado[puntos_en_rango]
    
    axes[0, 1].scatter(tiempos_puntos, amplitudes_puntos, 
                      color='red', s=30, zorder=5, label=f'Puntos {metodo}')
    axes[0, 1].set_title(f'Detección de Periodos ({metodo})')
    axes[0, 1].set_xlabel('Tiempo (s)')
    axes[0, 1].set_ylabel('Amplitud')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Distribución de periodos
    axes[1, 0].hist(periodos * 1000, bins=15, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].axvline(np.mean(periodos) * 1000, color='red', linestyle='--', 
                      label=f'Promedio: {np.mean(periodos)*1000:.3f} ms')
    axes[1, 0].set_title('Distribución de Periodos')
    axes[1, 0].set_xlabel('Periodo (ms)')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gráfico 4: Evolución temporal de periodos
    axes[1, 1].plot(periodos * 1000, 'o-', markersize=4, linewidth=1)
    axes[1, 1].axhline(np.mean(periodos) * 1000, color='red', linestyle='--', 
                      label='Promedio')
    axes[1, 1].set_title('Evolución Temporal de Periodos')
    axes[1, 1].set_xlabel('Número de Periodo')
    axes[1, 1].set_ylabel('Periodo (ms)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Procesar todos los audios
print("INICIANDO ANÁLISIS DE JITTER PARA 6 AUDIOS")
print("="*70)

resultados = []

for nombre, ruta in rutas_audios.items():
    # Determinar género basado en el nombre
    if 'fem' in nombre.lower():
        genero = 'femenino'
    else:
        genero = 'masculino'
    
    # Verificar si el archivo existe
    if not os.path.exists(ruta):
        print(f"\n ARCHIVO NO ENCONTRADO: {ruta}")
        continue
    
    # Procesar audio
    jitter_abs, jitter_rel, periodo_prom, f0, n_periodos = procesar_audio(
        ruta, nombre, genero)
    
    if jitter_abs is not None:
        resultados.append({
            'Audio': nombre,
            'Género': genero,
            'Jitter Absoluto (ms)': jitter_abs * 1000,
            'Jitter Relativo (%)': jitter_rel,
            'Periodo Promedio (ms)': periodo_prom * 1000,
            'F0 Promedio (Hz)': f0,
            'N° Periodos': n_periodos
        })

# Mostrar tabla resumen
print("\n" + "="*80)
print("RESUMEN GENERAL DE RESULTADOS")
print("="*80)

if resultados:
    df_resultados = pd.DataFrame(resultados)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 6)
    
    print("\nTabla de Resultados:")
    print(df_resultados.to_string(index=False))
    
    # Estadísticas por género
    print("\n" + "-"*50)
    print("ESTADÍSTICAS POR GÉNERO")
    print("-"*50)
    
    stats_femenino = df_resultados[df_resultados['Género'] == 'femenino']
    stats_masculino = df_resultados[df_resultados['Género'] == 'masculino']
    
    if not stats_femenino.empty:
        print("\n🔹 VOCES FEMENINAS:")
        print(f"Jitter Relativo Promedio: {stats_femenino['Jitter Relativo (%)'].mean():.4f} %")
        print(f"F0 Promedio: {stats_femenino['F0 Promedio (Hz)'].mean():.2f} Hz")
    
    if not stats_masculino.empty:
        print("\n🔹 VOCES MASCULINAS:")
        print(f"Jitter Relativo Promedio: {stats_masculino['Jitter Relativo (%)'].mean():.4f} %")
        print(f"F0 Promedio: {stats_masculino['F0 Promedio (Hz)'].mean():.2f} Hz")
    
    # Guardar resultados en CSV
    df_resultados.to_csv('/content/drive/Shareddrives/Labs procesamiento de señales/Lab 3/resultados_jitter.csv', index=False)
    print(f"\n Resultados guardados en: resultados_jitter.csv")
    
else:
    print(" No se pudieron procesar ninguno de los audios")

print("\n ANÁLISIS COMPLETADO")

```

Obtendiendose los siguientes resultados para cada sujeto.

Sujeto 1 Femenino
============================================================
Frecuencia de muestreo: 48000 Hz

Duración: 4.08 segundos

Filtro aplicado: 150-500 Hz

Periodos detectados - Cruces por cero: 1299

Periodos detectados - Picos: 361

--- RESULTADOS JITTER ---

Jitter absoluto: 0.803030 ms

Jitter relativo: 26.409212 %

Periodo promedio: 3.040720 ms

Frecuencia fundamental (F0): 328.87 Hz

Número de periodos analizados: 1299

<img width="1489" height="985" alt="image" src="https://github.com/user-attachments/assets/df711756-c8f4-4bab-b6d7-16871956955e" />


Sujeto 2 Femenino
============================================================
Frecuencia de muestreo: 48000 Hz

Duración: 3.56 segundos

Filtro aplicado: 150-500 Hz

Periodos detectados - Cruces por cero: 1043

Periodos detectados - Picos: 298

--- RESULTADOS JITTER ---

Jitter absoluto: 1.031830 ms

Jitter relativo: 32.868550 %

Periodo promedio: 3.139262 ms

Frecuencia fundamental (F0): 318.55 Hz

Número de periodos analizados: 1043


<img width="1489" height="985" alt="image" src="https://github.com/user-attachments/assets/b7815432-91c1-4a39-9b9b-b8b16a33d38c" />


Sujeto 3 Femenino
============================================================
Frecuencia de muestreo: 48000 Hz

Duración: 4.24 segundos

Filtro aplicado: 150-500 Hz

Periodos detectados - Cruces por cero: 1302

Periodos detectados - Picos: 462

--- RESULTADOS JITTER ---

Jitter absoluto: 0.666507 ms

Jitter relativo: 21.102489 %

Periodo promedio: 3.158426 ms

Frecuencia fundamental (F0): 316.61 Hz

Número de periodos analizados: 1302


<img width="1489" height="985" alt="image" src="https://github.com/user-attachments/assets/7b356db2-5ce0-4ea3-bec0-2d1b14639c82" />


Sujeto 1 Masculino
============================================================

Frecuencia de muestreo: 48000 Hz

Duración: 5.04 segundos

Filtro aplicado: 80-400 Hz

Periodos detectados - Cruces por cero: 1044

Periodos detectados - Picos: 488

--- RESULTADOS JITTER ---

Jitter absoluto: 1.687880 ms

Jitter relativo: 35.818103 %

Periodo promedio: 4.712364 ms

Frecuencia fundamental (F0): 212.21 Hz

Número de periodos analizados: 1044


<img width="1490" height="985" alt="image" src="https://github.com/user-attachments/assets/9f1e6c09-db9a-4b30-a34e-0b5c03282385" />


Sujeto 2 Masculino
============================================================

Frecuencia de muestreo: 48000 Hz

Duración: 3.80 segundos

Filtro aplicado: 80-400 Hz

Periodos detectados - Cruces por cero: 851

Periodos detectados - Picos: 367

--- RESULTADOS JITTER ---

Jitter absoluto: 1.754755 ms

Jitter relativo: 40.845554 %

Periodo promedio: 4.296073 ms

Frecuencia fundamental (F0): 232.77 Hz

Número de periodos analizados: 851


<img width="1489" height="985" alt="image" src="https://github.com/user-attachments/assets/d7c5587a-2932-4444-b23a-5d80a7bb8c0a" />


Sujeto 3 Masculino
============================================================

Frecuencia de muestreo: 48000 Hz

Duración: 3.56 segundos

Filtro aplicado: 80-400 Hz

Periodos detectados - Cruces por cero: 821

Periodos detectados - Picos: 340

--- RESULTADOS JITTER ---

Jitter absoluto: 1.677998 ms

Jitter relativo: 40.084710 %

Periodo promedio: 4.186130 ms

Frecuencia fundamental (F0): 238.88 Hz

Número de periodos analizados: 821


<img width="1489" height="985" alt="image" src="https://github.com/user-attachments/assets/1ba68e16-fd92-473e-9b2f-87c72cb8bb70" />


 Medición del Shimmer (variación en la amplitud):

   Detecte los picos de amplitud Ai en cada ciclo.
  
   Obtenga el shimmer absoluto: 


  <img width="313" height="90" alt="image" src="https://github.com/user-attachments/assets/994db285-de06-408b-827f-66651a5cb0a1" />


   Calcule el shimmer relativo (%): 

  
  <img width="339" height="78" alt="image" src="https://github.com/user-attachments/assets/bc624b39-5211-4d98-9bf9-40f5343dba37" />


A continuación, se muestra el codigo que se realizó para calcular cada una de las mediciones de shimer para cada sujeto femenino y masculino.

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd
from google.colab import drive
import os
çdrive.mount('/content/drive')

rutas_audios = {
    'Sujeto 1 Femenino': '/content/drive/Shareddrives/Labs procesamiento de señales/Lab 3/sujeto 1 fem.wav',
    'Sujeto 2 Femenino': '/content/drive/Shareddrives/Labs procesamiento de señales/Lab 3/sujeto 2 fem.wav',
    'Sujeto 3 Femenino': '/content/drive/Shareddrives/Labs procesamiento de señales/Lab 3/sujeto 3 fem.wav',
    'Sujeto 1 Masculino': '/content/drive/Shareddrives/Labs procesamiento de señales/Lab 3/sujeto masc 1 1.wav',
    'Sujeto 2 Masculino': '/content/drive/Shareddrives/Labs procesamiento de señales/Lab 3/sujeto 2 masc.wav',
    'Sujeto 3 Masculino': '/content/drive/Shareddrives/Labs procesamiento de señales/Lab 3/sujeto 3 masc.wav'
}

# Función para aplicar filtro pasa banda 
def filtro_pasabanda(audio, fs, lowcut, highcut, order=4):
    """Aplica filtro pasa banda Butterworth"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Verificar longitud del audio
    min_length_required = 3 * (order + 1)
    if len(audio) <= min_length_required:
        new_order = max(1, (len(audio) // 3) - 1)
        order = new_order

    if order <= 0:
        return np.zeros_like(audio)

    b, a = butter(order, [low, high], btype='band')
    audio_filtrado = filtfilt(b, a, audio)
    return audio_filtrado

# Función para detectar amplitudes usando picos
def detectar_amplitudes(audio, fs, distancia_minima=None):
    """Detecta amplitudes (picos) en cada ciclo de la señal"""
    if distancia_minima is None:
        distancia_minima = int(0.005 * fs)  # 5ms mínima distancia entre picos

    # Encontrar picos positivos
    picos, propiedades = find_peaks(audio, distance=distancia_minima, height=0.1*np.max(np.abs(audio)))
    amplitudes = propiedades['peak_heights']

    # Filtrar amplitudes válidas (eliminar outliers)
    amplitudes_filtradas = []
    picos_filtrados = []

    for i in range(len(amplitudes)):
        # Filtrar amplitudes que estén dentro de un rango razonable
        if amplitudes[i] > 0.05 * np.max(amplitudes):  # Al menos 5% de la amplitud máxima
            amplitudes_filtradas.append(amplitudes[i])
            picos_filtrados.append(picos[i])

    return np.array(amplitudes_filtradas), np.array(picos_filtrados)

# Función para calcular shimmer
def calcular_shimmer(amplitudes):
    """Calcula shimmer absoluto y relativo según las fórmulas especificadas"""
    if len(amplitudes) < 2:
        return 0, 0, 0, 0

    N = len(amplitudes)

    # Calcular shimmer absoluto
    # Shimmer_abs = (1/(N-1)) * Σ|A_i - A_{i+1}|
    suma_diferencias = 0
    for i in range(N-1):
        suma_diferencias += abs(amplitudes[i] - amplitudes[i+1])

    shimmer_abs = suma_diferencias / (N-1)

    # Calcular amplitud promedio
    # A_promedio = (1/N) * ΣA_i
    amplitud_promedio = np.mean(amplitudes)

    # Calcular shimmer relativo (%)
    # Shimmer_rel = (Shimmer_abs / A_promedio) * 100
    shimmer_rel = (shimmer_abs / amplitud_promedio) * 100

    # Shimmer local (variación entre amplitudes consecutivas)
    shimmer_local = np.mean(np.abs(np.diff(amplitudes)))

    return shimmer_abs, shimmer_rel, amplitud_promedio, shimmer_local

# Función principal para procesar un audio
def procesar_audio(ruta_audio, nombre_audio, genero='femenino'):
    """Procesa un archivo de audio y calcula sus métricas de shimmer"""
    print(f"\n{'='*60}")
    print(f"PROCESANDO: {nombre_audio}")
    print(f"{'='*60}")

    try:
        # Cargar audio
        fs, audio = wavfile.read(ruta_audio)
        print(f"Frecuencia de muestreo: {fs} Hz")
        print(f"Duración: {len(audio)/fs:.2f} segundos")

        # Convertir a mono si es stereo
        if len(audio.shape) > 1:
            audio = audio[:, 0]
            print("Audio convertido a mono")

        # Convertir a float y normalizar
        audio = audio.astype(np.float32)
        audio = audio / np.max(np.abs(audio))

        # Aplicar filtro pasa banda según el género
        if genero.lower() == 'femenino':
            audio_filtrado = filtro_pasabanda(audio, fs, 150, 500)
            rango_frecuencia = "150-500 Hz"
        else:
            audio_filtrado = filtro_pasabanda(audio, fs, 80, 400)
            rango_frecuencia = "80-400 Hz"

        print(f"Filtro aplicado: {rango_frecuencia}")

        # Detectar amplitudes
        amplitudes, picos = detectar_amplitudes(audio_filtrado, fs)

        print(f"Amplitudes detectadas: {len(amplitudes)}")

        if len(amplitudes) < 2:
            print("ADVERTENCIA: No se detectaron suficientes amplitudes para calcular shimmer")
            return None, None, None, None

        # Calcular métricas de shimmer
        shimmer_abs, shimmer_rel, amplitud_promedio, shimmer_local = calcular_shimmer(amplitudes)

        print(f"\n--- RESULTADOS SHIMMER ---")
        print(f"Shimmer absoluto: {shimmer_abs:.6f}")
        print(f"Shimmer relativo: {shimmer_rel:.6f} %")
        print(f"Shimmer local: {shimmer_local:.6f}")
        print(f"Amplitud promedio: {amplitud_promedio:.6f}")
        print(f"Número de amplitudes analizadas: {len(amplitudes)}")

        # Mostrar fórmulas aplicadas
        print(f"\n--- FÓRMULAS APLICADAS ---")
        print(f"Shimmer_abs = (1/(N-1)) * Σ|A_i - A_i+1| = {shimmer_abs:.6f}")
        print(f"Shimmer_rel = (Shimmer_abs / A_promedio) * 100 = {shimmer_rel:.6f}%")
        print(f"Donde A_promedio = {amplitud_promedio:.6f}")

        # Visualización
        visualizar_resultados(audio, audio_filtrado, picos, amplitudes, nombre_audio, fs)

        return shimmer_abs, shimmer_rel, amplitud_promedio, len(amplitudes)

    except Exception as e:
        print(f"ERROR procesando {nombre_audio}: {e}")
        return None, None, None, None

# Función para visualizar resultados
def visualizar_resultados(audio_original, audio_filtrado, picos, amplitudes, nombre_audio, fs):
    """Genera gráficos para visualizar los resultados de shimmer"""
    # Mostrar solo una parte del audio para mejor visualización
    muestras_visualizar = min(5000, len(audio_original))
    tiempo = np.arange(muestras_visualizar) / fs

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Análisis de Shimmer - {nombre_audio}', fontsize=16)

    # Gráfico 1: Señal original vs filtrada con picos detectados
    axes[0, 0].plot(tiempo, audio_original[:muestras_visualizar],
                   color='gray', alpha=0.7, label='Original')
    axes[0, 0].plot(tiempo, audio_filtrado[:muestras_visualizar],
                   color='blue', linewidth=1, label='Filtrada')

    # Marcar picos en el rango visualizado
    picos_en_rango = [p for p in picos if p < muestras_visualizar]
    tiempos_picos = np.array(picos_en_rango) / fs
    amplitudes_picos = audio_filtrado[picos_en_rango]

    axes[0, 0].scatter(tiempos_picos, amplitudes_picos,
                      color='red', s=50, zorder=5, label='Picos detectados (A_i)')

    axes[0, 0].set_title('Señal con Detección de Picos de Amplitud')
    axes[0, 0].set_xlabel('Tiempo (s)')
    axes[0, 0].set_ylabel('Amplitud')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Gráfico 2: Distribución de amplitudes
    axes[0, 1].hist(amplitudes, bins=15, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].axvline(np.mean(amplitudes), color='red', linestyle='--',
                      linewidth=2, label=f'Promedio: {np.mean(amplitudes):.3f}')
    axes[0, 1].set_title('Distribución de Amplitudes (A_i)')
    axes[0, 1].set_xlabel('Amplitud')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Gráfico 3: Evolución temporal de amplitudes
    axes[1, 0].plot(amplitudes, 'o-', markersize=5, linewidth=1.5, color='orange')
    axes[1, 0].axhline(np.mean(amplitudes), color='red', linestyle='--',
                      linewidth=2, label='Amplitud promedio')

    # Resaltar las diferencias entre amplitudes consecutivas
    for i in range(len(amplitudes)-1):
        axes[1, 0].plot([i, i+1], [amplitudes[i], amplitudes[i+1]],
                       'r-', alpha=0.3, linewidth=0.5)

    axes[1, 0].set_title('Evolución Temporal de Amplitudes\n(Líneas rojas: |A_i - A_i+1|)')
    axes[1, 0].set_xlabel('Índice de Amplitud (i)')
    axes[1, 0].set_ylabel('Amplitud (A_i)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Gráfico 4: Diferencias entre amplitudes consecutivas
    diferencias = np.abs(np.diff(amplitudes))
    axes[1, 1].plot(diferencias, 's-', markersize=4, linewidth=1, color='purple')
    axes[1, 1].axhline(np.mean(diferencias), color='red', linestyle='--',
                      linewidth=2, label=f'Promedio: {np.mean(diferencias):.3f}')
    axes[1, 1].set_title('Diferencias |A_i - A_i+1|\n(Contribución al Shimmer)')
    axes[1, 1].set_xlabel('Índice de Diferencia')
    axes[1, 1].set_ylabel('|A_i - A_i+1|')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Procesar todos los audios
print("INICIANDO ANÁLISIS DE SHIMMER PARA 6 AUDIOS")
print("="*70)

resultados = []

for nombre, ruta in rutas_audios.items():
    # Determinar género basado en el nombre
    if 'fem' in nombre.lower():
        genero = 'femenino'
    else:
        genero = 'masculino'

    # Verificar si el archivo existe
    if not os.path.exists(ruta):
        print(f"\n ARCHIVO NO ENCONTRADO: {ruta}")
        continue

    # Procesar audio
    shimmer_abs, shimmer_rel, amplitud_prom, n_amplitudes = procesar_audio(ruta, nombre, genero)

    if shimmer_abs is not None:
        resultados.append({
            'Audio': nombre,
            'Género': genero,
            'Shimmer Absoluto': shimmer_abs,
            'Shimmer Relativo (%)': shimmer_rel,
            'Amplitud Promedio': amplitud_prom,
            'N° Amplitudes': n_amplitudes
        })

# Mostrar tabla resumen
print("\n" + "="*80)
print("RESUMEN GENERAL DE RESULTADOS - SHIMMER")
print("="*80)

if resultados:
    df_resultados = pd.DataFrame(resultados)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 6)

    print("\nTabla de Resultados:")
    print(df_resultados.to_string(index=False))

    # Estadísticas por género
    print("\n" + "-"*50)
    print("ESTADÍSTICAS POR GÉNERO")
    print("-"*50)

    stats_femenino = df_resultados[df_resultados['Género'] == 'femenino']
    stats_masculino = df_resultados[df_resultados['Género'] == 'masculino']

    if not stats_femenino.empty:
        print("\n🔹 VOCES FEMENINAS:")
        print(f"Shimmer Relativo Promedio: {stats_femenino['Shimmer Relativo (%)'].mean():.4f} %")
        print(f"Shimmer Absoluto Promedio: {stats_femenino['Shimmer Absoluto'].mean():.6f}")
        print(f"Amplitud Promedio: {stats_femenino['Amplitud Promedio'].mean():.6f}")
        print(f"Número de sujetos: {len(stats_femenino)}")

    if not stats_masculino.empty:
        print("\n🔹 VOCES MASCULINAS:")
        print(f"Shimmer Relativo Promedio: {stats_masculino['Shimmer Relativo (%)'].mean():.4f} %")
        print(f"Shimmer Absoluto Promedio: {stats_masculino['Shimmer Absoluto'].mean():.6f}")
        print(f"Amplitud Promedio: {stats_masculino['Amplitud Promedio'].mean():.6f}")
        print(f"Número de sujetos: {len(stats_masculino)}")

    # Análisis comparativo
    print("\n" + "-"*50)
    print("ANÁLISIS COMPARATIVO")
    print("-"*50)

    if not stats_femenino.empty and not stats_masculino.empty:
        shimmer_rel_fem = stats_femenino['Shimmer Relativo (%)'].mean()
        shimmer_rel_masc = stats_masculino['Shimmer Relativo (%)'].mean()

        print(f"Diferencia en Shimmer Relativo: {abs(shimmer_rel_fem - shimmer_rel_masc):.4f} %")
        if shimmer_rel_fem > shimmer_rel_masc:
            print("Las voces femeninas presentan mayor shimmer que las masculinas")
        else:
            print("Las voces masculinas presentan mayor shimmer que las femeninas")

    # Guardar resultados en CSV
    df_resultados.to_csv('/content/drive/Shareddrives/Labs procesamiento de señales/Lab 3/resultados_shimmer.csv', index=False)
    print(f"\n Resultados guardados en: resultados_shimmer.csv")

else:
    print(" No se pudieron procesar ninguno de los audios")

print("\n ANÁLISIS DE SHIMMER COMPLETADO")

```

Obteniéndose los siguientes resultados:

Sujeto 1 Femenino
============================================================

Frecuencia de muestreo: 48000 Hz

Duración: 4.08 segundos

Filtro aplicado: 150-500 Hz

Amplitudes detectadas: 364

--- RESULTADOS SHIMMER ---

Shimmer absoluto: 0.045846

Shimmer relativo: 24.449685 %

Amplitud promedio: 0.187513

Número de amplitudes analizadas: 364

Donde A_promedio = 0.187513

<img width="1490" height="985" alt="image" src="https://github.com/user-attachments/assets/37461180-11a8-4e40-9e45-844cfdb1ac00" />

Sujeto 2 Femenino
============================================================

Frecuencia de muestreo: 48000 Hz

Duración: 3.56 segundos


Filtro aplicado: 150-500 Hz

Amplitudes detectadas: 303

--- RESULTADOS SHIMMER ---

Shimmer absoluto: 0.038508

Shimmer relativo: 27.511857 %

Amplitud promedio: 0.139968

Número de amplitudes analizadas: 303

Donde A_promedio = 0.139968


<img width="1489" height="985" alt="image" src="https://github.com/user-attachments/assets/d55cf136-6592-4e74-9f34-a3f5e464f488" />


Sujeto 3 Femenino
============================================================

Frecuencia de muestreo: 48000 Hz

Duración: 4.24 segundos

Filtro aplicado: 150-500 Hz

Amplitudes detectadas: 450

--- RESULTADOS SHIMMER ---

Shimmer absoluto: 0.025719

Shimmer relativo: 30.384430 %

Shimmer local: 0.025719

Amplitud promedio: 0.084646

Número de amplitudes analizadas: 450

Donde A_promedio = 0.084646


<img width="1487" height="985" alt="image" src="https://github.com/user-attachments/assets/be013df0-d6ec-4e91-9455-ac4666cb7770" />


Sujeto 1 Masculino
============================================================

Frecuencia de muestreo: 48000 Hz

Duración: 5.04 segundos

Filtro aplicado: 80-400 Hz

Amplitudes detectadas: 464

--- RESULTADOS SHIMMER ---

Shimmer absoluto: 0.021168

Shimmer relativo: 18.913645 %

Amplitud promedio: 0.111920

Número de amplitudes analizadas: 464

Donde A_promedio = 0.111920


<img width="1487" height="985" alt="image" src="https://github.com/user-attachments/assets/8d2c2892-d0bc-4996-8340-470dee468099" />


Sujeto 2 Masculino
============================================================

Frecuencia de muestreo: 48000 Hz

Duración: 3.80 segundos

Filtro aplicado: 80-400 Hz

Amplitudes detectadas: 356

--- RESULTADOS SHIMMER ---

Shimmer absoluto: 0.040427

Shimmer relativo: 22.495124 %

Shimmer local: 0.040427

Amplitud promedio: 0.179716

Número de amplitudes analizadas: 356

Donde A_promedio = 0.179716


<img width="1489" height="985" alt="image" src="https://github.com/user-attachments/assets/9dd4892c-3064-4a88-8c33-fe47a27c9d54" />


Sujeto 3 Masculino
============================================================

Frecuencia de muestreo: 48000 Hz

Duración: 3.56 segundos

Audio convertido a mono

Filtro aplicado: 80-400 Hz

Amplitudes detectadas: 340

--- RESULTADOS SHIMMER ---

Shimmer absoluto: 0.041566

Shimmer relativo: 33.591883 %

Amplitud promedio: 0.123739

Número de amplitudes analizadas: 340

Donde A_promedio = 0.123739


<img width="1489" height="985" alt="image" src="https://github.com/user-attachments/assets/a9939cdf-129e-4b44-abcb-bbd3a70df175" />



 Presente los valores obtenidos de jitter y shimmer para cada una de las 6
grabaciones (3 hombres, 3 mujeres). 

Datos obtenidos jitter:

<img width="999" height="354" alt="image" src="https://github.com/user-attachments/assets/7b1aaf13-0266-4923-99fe-5691f464f66d" />


Dados Obtenidos shimmer:

<img width="681" height="450" alt="image" src="https://github.com/user-attachments/assets/5ba4d3ed-200f-4cd7-abe9-310e716414ed" />



PARTE C – Comparación y conclusiones 
Para realizar la parte C se necesitan conocimientos previos para entender los resultados, por eso tendremos un mini glosario con lo mas importante antes de empezar con las comparaciones y conclusiones.


**Shimmer:** su traducción directa es brillo, en el contexto en el que hablamos (voces) este brillo se refiere a una variación de manera porcentual en la amplitud de la onda sonora de la voz, estas variaciones se pueden percibir como ronquera o asperezas de la voz. Y es calculada como la diferencia absoluta promedio entre las amplitudes de periodos consecutivos dividida entre la amplitud media.


**Jitter:** Este se refiere a las pequeñas variaciones de las frecuencias fundamentales entre ciclos glóticos, e indica la estabilidad de la voz.


**Ciclos glóticos:** Ciclo de apertura y cierre de la glotis (cuerdas vocales) que generan sonidos.


__Espectro de frecuencia:__ Representación de frecuencias que componen una onda en este caso una onda sonora, en el caso de la voz humana se sitúa principalmente entre 80 Hz y 1100 Hz aunque también puede haber variaciones.


**El nivel de RMS:** El Root Mean Square o "media cuadrática" es el nivel de volumen promedio y se mide en decibeles 

**Frecuencia fundamental:** Tasa de vibración de las cuerdas vocales al general los sonidos, Se mide utilizando herramientas acústicas y se representa como F0


***Comparar los resultados obtenidos entre las voces masculinas y femeninas.***
<img width="925" height="525" alt="image" src="https://github.com/user-attachments/assets/d2d51650-033d-4a57-aafd-a738b761f12b" />


*** ¿Qué diferencias se observan en la frecuencia fundamental?***
Como se mencionó antes la frecuencia fundamental es la tasa de vibración de la voz y tiene un rango normal de entre 85 y 180 Hz, con un promedio de alrededor de 120 Hz en hombres adultos y en mujeres adultas entre 165 y 255 Hz, con un promedio de alrededor de 210 Hz. 
Estas diferencias son debido a factores anatómicos especialmente del aparato fonador, donde en las mujeres se evidencia unas cuerdas vocales más cortas (12–20 mm) y más delgadas, lo que produce una vibración más rápida, mientras que en los hombres tienen una laringe más grande y sus hormonas como la testosterona aumenta la masa y longitud de las cuerdas, lo que disminuye la frecuencia fundamental.
Hablando específicamente de nuestros resultados obtenidos tenemos concordancia con lo antes mencionado, las voces femeninas obtuvieron una F0 promedio de 321,34 mientras que las masculinas mostraron un promedio de 227.95 Hz, confirmando así que las voces femeninas son más agudas y presentan mayor vibración que las masculinas.


*** ¿Qué otras diferencias notan en términos de brillo, media o intensidad?***
Respecto al brillo, media o intensidad se evidencio en los resultados que las voces de los sujetos femeninos tuvieron un shimmer relativo mayor que el de los hombres (femenino 27,45%, masculino 25%), lo que nos muestra una mayor inestabilidad de parte de los hombres en la frecuencia de sus ciclos glóticos, en cambio los sujetos femeninos mostraron una mayor variabilidad de sus amplitudes mostrando así una voz más brillante. En cuanto a la amplitud promedio se obtuvieron valores similares lo que muestra que no hubo gran variación de la intensidad de las voces entre géneros 

*** Redactar conclusiones sobre el comportamiento de la voz en hombres y
mujeres a partir de los análisis realizados.***

En conclusión, los datos obtenidos concuerdan con los datos teóricos y fisiológicos esperados, donde no es difícil evidenciar las diferencias entre voces femeninas y masculinas desde que las voces femeninas cuentan con más brillo y una frecuencia mayor y su variabilidad, demostrando así que las muestras fueron tomadas de la mejor manera y su posterior análisis fue bueno. Estos resultados son los esperados gracias a las anteriores investigaciones y el conocimiento fisiológico adquirido atreves de la carrera e investigaciones. 


*** Discuta la importancia clínica del jitter y shimmer en el análisis de la voz.***





















