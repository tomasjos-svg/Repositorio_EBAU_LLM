# Aplicación de la actualización en Windows

1. Descomprima este paquete.
2. Copie **todo su contenido** dentro de la carpeta local `Repositorio_EBAU_LLM` que usa GitHub Desktop. Acepte combinar carpetas y reemplazar los archivos de documentación si Windows lo pregunta.
3. Abra esa carpeta en el Explorador de Windows.
4. Haga clic en la barra de direcciones, escriba `cmd` y pulse Intro.
5. Ejecute:

```cmd
python PREPARAR_REPOSITORIO.py
```

6. Espere a que termine. El cálculo de hashes puede durar varios minutos.
7. Si el informe indica posibles secretos, **no haga push** hasta revocar la clave y eliminarla. Consulte `metadata/SECURITY_SCAN.csv`.
8. Abra GitHub Desktop. Escriba como resumen del commit: `Prepare scientific repository for reproducible publication`.
9. Pulse **Commit to main** y después **Push origin**.

El proceso conserva todos los materiales. Solo reorganiza las carpetas originales y genera documentación e inventarios.
