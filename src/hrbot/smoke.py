import sys
import time

from hrbot.config import Settings
from hrbot.llm import get_llm


def smoke(settings: Settings | None = None) -> None:
    """Send a test prompt to the LLM, print reply and latency, or exit with a clear hint."""
    try:
        llm = get_llm(settings=settings)
        start = time.perf_counter()
        response = llm.invoke("Reply with exactly: OK")
        latency = time.perf_counter() - start
        print(f"Reply: {response.content}")
        print(f"Latency: {latency:.2f}s")
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
        print("Hint: Is GROQ_API_KEY valid and reachable? Is LLM_MODEL valid?")
        sys.exit(1)


if __name__ == "__main__":
    smoke()
