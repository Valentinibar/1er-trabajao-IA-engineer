"""
Ejercicio Unidad 1 - Orquestador concurrente de modelos
Requiere Python 3.12+ (usa asyncio.timeout y sintaxis moderna de tipos).

Parte 1: consulta 3 "modelos" en simultáneo con asyncio.gather,
         con un timeout TOTAL de 2 segundos.
Parte 2: dispara 10 llamadas simuladas, pero un asyncio.Semaphore(2)
         garantiza que solo 2 se ejecuten al mismo tiempo.

No usa time.sleep (bloqueante): toda la latencia se simula con await asyncio.sleep().
"""

import asyncio
import logging
import random
import sys
import time

# ---------------------------------------------------------------------------
# Logging con tiempo relativo para ver claramente cuándo arranca cada tarea
# ---------------------------------------------------------------------------
T0 = time.perf_counter()


class RelativeTimeFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.reltime = f"{time.perf_counter() - T0:6.3f}s"
        return super().format(record)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(RelativeTimeFormatter("[%(reltime)s] %(message)s"))
log = logging.getLogger("orquestador")
log.addHandler(handler)
log.setLevel(logging.INFO)

TIMEOUT_TOTAL = 2.0     # segundos para toda la Parte 1
MAX_CONCURRENTES = 2    # límite del semáforo en la Parte 2
TOTAL_LLAMADAS = 10     # llamadas a disparar en la Parte 2


# ---------------------------------------------------------------------------
# Paso 2 - Simulación de llamadas a modelos (I/O-bound)
# ---------------------------------------------------------------------------
async def gpt_4_call(prompt: str) -> str:
    log.info("gpt_4_call       -> inicio")
    await asyncio.sleep(1.2)                  # latencia simulada
    log.info("gpt_4_call       -> fin")
    return f"[GPT-4] respuesta a: {prompt!r}"


async def claude_3_call(prompt: str) -> str:
    log.info("claude_3_call    -> inicio")
    await asyncio.sleep(0.8)
    log.info("claude_3_call    -> fin")
    return f"[Claude 3] respuesta a: {prompt!r}"


async def local_llama_call(prompt: str) -> str:
    log.info("local_llama_call -> inicio")
    await asyncio.sleep(3.0)                  # más lento que el timeout a propósito
    log.info("local_llama_call -> fin")
    return f"[Llama local] respuesta a: {prompt!r}"


# ---------------------------------------------------------------------------
# Pasos 3 y 4 - Orquestación con gather + timeout total de 2 segundos
# ---------------------------------------------------------------------------
async def consultar_modelos(prompt: str) -> dict[str, str]:
    # Creamos Tasks explícitas para poder ver, si hay timeout,
    # cuáles terminaron y cuáles quedaron colgadas.
    tareas = {
        "gpt-4": asyncio.create_task(gpt_4_call(prompt)),
        "claude-3": asyncio.create_task(claude_3_call(prompt)),
        "llama-local": asyncio.create_task(local_llama_call(prompt)),
    }

    try:
        async with asyncio.timeout(TIMEOUT_TOTAL):
            await asyncio.gather(*tareas.values(), return_exceptions=True)
    except TimeoutError:
        # gather cancela las tareas pendientes; las ya terminadas conservan su resultado
        log.error(f"TIMEOUT: la ejecución total superó {TIMEOUT_TOTAL}s. "
                  "Se usan las respuestas disponibles (fallback).")

    resultados: dict[str, str] = {}
    for nombre, tarea in tareas.items():
        if tarea.cancelled():
            resultados[nombre] = "ERROR: cancelado por timeout"
        elif tarea.exception() is not None:
            resultados[nombre] = f"ERROR: {tarea.exception()!r}"
        else:
            resultados[nombre] = tarea.result()
    return resultados


# ---------------------------------------------------------------------------
# Paso 5 - Control de flujo con Semaphore (10 llamadas, máx. 2 a la vez)
# ---------------------------------------------------------------------------
activas = 0
pico_concurrencia = 0


async def llamada_limitada(i: int, sem: asyncio.Semaphore) -> str:
    global activas, pico_concurrencia
    log.info(f"  llamada {i:02d} -> encolada")          # todas aparecen casi al mismo tiempo
    async with sem:
        activas += 1
        pico_concurrencia = max(pico_concurrencia, activas)
        log.info(f"  llamada {i:02d} -> EJECUTANDO (activas: {activas})")
        await asyncio.sleep(random.uniform(0.3, 0.8))
        activas -= 1
        log.info(f"  llamada {i:02d} -> terminada")
    return f"respuesta_{i:02d}"


async def rafaga_con_semaforo() -> list[str]:
    sem = asyncio.Semaphore(MAX_CONCURRENTES)
    return await asyncio.gather(
        *(llamada_limitada(i, sem) for i in range(1, TOTAL_LLAMADAS + 1))
    )


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
async def main() -> None:
    log.info("=== PARTE 1: 3 modelos en paralelo (timeout total 2s) ===")
    inicio = time.perf_counter()
    resultados = await consultar_modelos("¿Qué es asyncio?")
    log.info(f"Parte 1 terminó en {time.perf_counter() - inicio:.2f}s")
    for modelo, respuesta in resultados.items():
        log.info(f"  {modelo:<12} => {respuesta}")

    log.info("")
    log.info(f"=== PARTE 2: {TOTAL_LLAMADAS} llamadas con Semaphore({MAX_CONCURRENTES}) ===")
    inicio = time.perf_counter()
    respuestas = await rafaga_con_semaforo()
    log.info(f"Parte 2 terminó en {time.perf_counter() - inicio:.2f}s: "
             f"{len(respuestas)} respuestas, pico de concurrencia = {pico_concurrencia}")
    assert pico_concurrencia <= MAX_CONCURRENTES, "¡El semáforo no limitó la concurrencia!"

    log.info("")
    log.info("Programa finalizado correctamente (el timeout no detuvo la ejecución).")


if __name__ == "__main__":
    if sys.version_info < (3, 12):
        sys.exit("Este ejercicio requiere Python 3.12 o superior.")
    asyncio.run(main())
