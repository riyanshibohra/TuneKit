# TuneKit CLI Manager

Herramienta de línea de comandos para gestionar los modelos soportados por TuneKit. Permite añadir, listar y eliminar modelos de la configuración sin necesidad de editar código.

## Uso Básico

El script se encuentra en `TuneKit/cli.py`. Ejecútalo desde la raíz del proyecto.

### Listar Modelos

Muestra todos los modelos configurados actualmente.

```bash
python TuneKit/cli.py list
```

### Añadir un Modelo

Inicia un asistente interactivo impulsado por "IA" que te guiará paso a paso.

```bash
python TuneKit/cli.py add
```

**Características Inteligentes:**
- **Autocompletado desde Hugging Face**: Al introducir el ID del modelo (ej: `google/gemma-3-270m-it`), la CLI consultará automáticamente la API de Hugging Face para obtener:
    - Ventana de contexto (Context Window).
    - Sugerencia de nombre amigable.
- **Validación**: Evita duplicados y campos vacíos.

El asistente te pedirá:
1. **Metadata**: ID, nombre, tamaño, GPU recomendada.
2. **Puntuación**: Qué tan bueno es el modelo para diferentes tareas (0-100).
3. **Razonamiento**: Características clave para mostrar al usuario.

### Eliminar un Modelo

Elimina un modelo existente por su clave única.

```bash
python TuneKit/cli.py remove <model_key>
```

Ejemplo:
```bash
python TuneKit/cli.py remove deepseek-1.3b
```

## Herramientas de Datos

### Enriquecimiento de Datos (Enrich)

Mejora automáticamente la calidad de tu dataset mediante métricas de calidad, filtrado y balanceo de clases.

```bash
python TuneKit/cli.py enrich <archivo.jsonl> [opciones]
```

**Opciones:**
- `--top_n <N>`: Mantiene solo los N mejores ejemplos según su puntuación de calidad.
- `--no-balance`: Desactiva el balanceo automático de clases (útil si no es una tarea de clasificación).
- `-o <archivo>`: Especifica el archivo de salida (por defecto: `nombre_enriched.jsonl`).

**Ejemplo:**
```bash
# Enriquecer y guardar solo los 100 mejores ejemplos
python TuneKit/cli.py enrich data.jsonl --top_n 100
```

## Archivos de Configuración

La configuración se almacena en `TuneKit/tunekit/data/models.json`. Este archivo es generado y gestionado automáticamente por el CLI, pero puede editarse manualmente si es necesario.

## Desarrollo

Si añades nuevos campos a la lógica de recomendación en `model_rec.py`, asegúrate de actualizar el CLI para soportarlos.
