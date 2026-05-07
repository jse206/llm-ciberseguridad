from __future__ import annotations

import importlib
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.schemas import ChatRequest, ChatResponse, UsageInfo

router = APIRouter(prefix="/api/chat", tags=["chat"])

_IMPL_DIR = Path(__file__).parent.parent.parent / "03_implementacion"
if str(_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(_IMPL_DIR))

_CHAINS = {
    "A": ("arquitectura_A_Text2SQL.chain", "ArchitectureAChain"),
    "B": ("arquitectura_B_API.chain", "ArchitectureBChain"),
    "C": ("arquitectura_C_GraphRAG.chain", "ArchitectureCChain"),
    "D": ("arquitectura_D_Toolformer.chain", "ArchitectureDChain"),
}

_ARCH_DESCRIPTIONS = {
    "A": "Text2SQL — consulta la BD NVD/CVE local mediante SQL generado por el LLM",
    "B": "API Calls — consulta NVD API v2.0 y AbuseIPDB en tiempo real",
    "C": "GraphRAG — razonamiento sobre el grafo MITRE ATT&CK",
    "D": "Toolformer — agente ReAct que elige entre A, B y C según la pregunta",
}


_CHAIN_INSTANCES: dict = {}


def _get_chain(arch: str):
    if arch not in _CHAINS:
        raise HTTPException(status_code=400, detail=f"Arquitectura '{arch}' no válida. Usa A, B, C o D.")
    if arch not in _CHAIN_INSTANCES:
        module_path, class_name = _CHAINS[arch]
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        _CHAIN_INSTANCES[arch] = cls()
    return _CHAIN_INSTANCES[arch]


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    chain = _get_chain(request.architecture)
    result = chain.run(request.question)

    usage_raw = result.get("usage", {})
    # Arch A stores per-step latencies; others store a single latency_s
    total_latency = sum(v for k, v in usage_raw.items() if k.endswith("_latency_s") and isinstance(v, (int, float)))
    if total_latency == 0:
        total_latency = usage_raw.get("latency_s", 0.0)
    usage = UsageInfo(
        total_tokens=usage_raw.get("total_tokens", 0),
        latency_s=round(total_latency, 3),
    )

    metadata = {k: v for k, v in result.items()
                if k not in ("answer", "hallucination_risk", "usage", "architecture")}

    return ChatResponse(
        answer=result.get("answer", ""),
        architecture=result.get("architecture", request.architecture),
        hallucination_risk=bool(result.get("hallucination_risk", False)),
        usage=usage,
        metadata=metadata,
    )


@router.get("/architectures")
def list_architectures() -> dict:
    return {"architectures": _ARCH_DESCRIPTIONS}
