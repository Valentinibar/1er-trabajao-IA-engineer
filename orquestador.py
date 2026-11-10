import asyncio
import time

inicio = time.perf_counter()


def log(mensaje: str) -> None:
    # Muestra el tiempo transcurrido para ver cuándo inicia cada tarea
    print(f"[{time.perf_counter() - inicio:.2f}s] {mensaje}")


# --- Simulación de llamadas a modelos (await asyncio.sleep, NO time.sleep) ---
async def gpt_4_call(prompt: str) -> str:
    log("gpt_4_call inicia")
    await asyncio.sleep(1.2)
    log("gpt_4_call termina")
    return f"GPT-4: respuesta a '{prompt}'"


async def claude_3_call(prompt: str) -> str:
    log("claude_3_call inicia")
    await asyncio.sleep(0.8)
    log("claude_3_call termina")
    return f"Claude 3: respuesta a '{prompt}'"


async def local_llama_call(prompt: str) -> str:
    log("local_llama_call inicia")
    await asyncio.sleep(3.0)  # tarda más que el timeout a propósito
    log("local_llama_call termina")
    return f"Llama local: respuesta a '{prompt}'"


# --- Orquestación con gather + timeout total de 2 segundos ---
async def consultar_modelos(prompt: str) -> None:
    try:
        async with asyncio.timeout(2.0):
            results = await asyncio.gather(
                gpt_4_call(prompt),
                claude_3_call(prompt),
                local_llama_call(prompt),
                return_exceptions=True,
            )
            for r in results:
                log(r)
    except TimeoutError:
        log("ERROR: los modelos tardaron más de 2 segundos. Aplicando fallback...")


# --- Control de flujo: 10 llamadas, solo 2 al mismo tiempo ---
sem = asyncio.Semaphore(2)


async def throttled_call(i: int) -> str:
    async with sem:
        log(f"llamada {i} inicia")
        await asyncio.sleep(0.5)
        log(f"llamada {i} termina")
        return f"respuesta {i}"


async def main():
    log("=== Parte 1: 3 modelos en simultáneo con timeout de 2s ===")
    await consultar_modelos("¿Qué es asyncio?")

    log("=== Parte 2: 10 llamadas con Semaphore(2) ===")
    results = await asyncio.gather(*[throttled_call(i) for i in range(1, 11)])
    log(f"Completadas: {len(results)} llamadas")

    log("Programa finalizado sin detenerse por el timeout")


if __name__ == "__main__":
    asyncio.run(main())
