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





















