import time
import logging
from typing import Tuple, Dict, Any, Optional

from config.settings import GROQ_API_KEY, GROQ_MODEL
from src.pipeline.feature_store import FeatureStoreManager

logger = logging.getLogger("LLMClient")


class LLMClient:
    """
    Robust LLM client wrapping Groq with millisecond latency tracking,
    automatic fallback to rule-based templates, and failure logging.
    """
    
    def __init__(self, api_key: str = GROQ_API_KEY, model: str = GROQ_MODEL):
        self.api_key = api_key
        self.model = model
        self.client = None
        self.fs = FeatureStoreManager()
        
        if self.api_key and self.api_key.strip():
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Groq client: {e}")

    def generate(
        self,
        prompt: str,
        system_instruction: str = "You are AgentWeather, an expert meteorological reasoning agent.",
        agent_name: str = "Agent",
        fallback_fn = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Sends prompt to LLM. If successful, logs and returns generated response.
        If LLM is unavailable or fails, executes fallback_fn, logs failure audit, and returns fallback text.
        """
        start_time = time.perf_counter()
        
        # 1. Attempt Groq API call if client is initialized
        if self.client:
            try:
                req_kwargs = dict(
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model,
                    temperature=0.4,
                    max_tokens=2048,
                    timeout=12.0,
                )
                if "gpt-oss" in self.model:
                    req_kwargs["reasoning_effort"] = "low"
                chat_completion = self.client.chat.completions.create(**req_kwargs)
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                response_text = (chat_completion.choices[0].message.content or "").strip()
                if not response_text:
                    raise ValueError("Groq returned empty content payload")
                
                # Log success
                self.fs.log_agent_call(
                    agent_name=agent_name,
                    prompt=prompt[:500],
                    response=response_text[:1000],
                    latency_ms=latency_ms,
                    status="SUCCESS",
                    model_used=self.model,
                    error_message=None
                )
                
                log_meta = {
                    "agent": agent_name,
                    "model": self.model,
                    "latency_ms": latency_ms,
                    "status": "SUCCESS"
                }
                return response_text, log_meta
                
            except Exception as e:
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                err_msg = str(e)
                logger.warning(f"Groq API call failed ({err_msg}). Switching to deterministic fallback.")
                
                # Log failure in Agent Call Log / Failure Log
                self.fs.log_agent_call(
                    agent_name=agent_name,
                    prompt=prompt[:500],
                    response="FALLBACK_USED",
                    latency_ms=latency_ms,
                    status="FAILED_API_ERROR",
                    model_used=self.model,
                    error_message=err_msg
                )
        
        # 2. Deterministic Fallback Execution
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        if fallback_fn:
            fallback_text = fallback_fn()
        else:
            fallback_text = "Forecast and alert synthesized using standard meteorological rule-based reasoning engine."
            
        status = "FALLBACK_NO_API_KEY" if not self.client else "FALLBACK_API_ERROR"
        
        if not self.client:
            self.fs.log_agent_call(
                agent_name=agent_name,
                prompt=prompt[:500],
                response=fallback_text[:1000],
                latency_ms=latency_ms,
                status=status,
                model_used="deterministic_rule_engine",
                error_message="No GROQ_API_KEY configured"
            )
            
        log_meta = {
            "agent": agent_name,
            "model": "deterministic_rule_engine",
            "latency_ms": latency_ms,
            "status": status
        }
        return fallback_text, log_meta

