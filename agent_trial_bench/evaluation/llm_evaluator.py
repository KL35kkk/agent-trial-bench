"""
LLM Evaluator for Agent Trial Bench
"""

import json
import re
import random
import os
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class LLMEvaluator(BaseEvaluator):
    """LLM evaluator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        config = config or {}
        self.model_name = config.get("model")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 2000)
        self.num_samples = max(1, int(config.get("num_samples", 1)))
        self.max_retries = max(0, int(config.get("max_retries", 2)))
        self.require_valid_json = config.get("require_valid_json", True)
        self.enforce_schema = config.get("enforce_schema", True)
        self.retry_invalid_json = config.get("retry_invalid_json", True)
        self.random_seed = config.get("random_seed")
        self._rng = random.Random(self.random_seed) if self.random_seed is not None else None
        self.trim_ratio = max(0.0, min(0.45, float(config.get("trim_ratio", 0.0))))
        self.enable_cache = bool(config.get("enable_cache", False))
        self.cache_dir = config.get("cache_dir", "output/llm_cache")

        self.client = None
        if config.get("client"):
            self.client = config["client"]
        elif OPENAI_AVAILABLE and config.get("api_key"):
            base_url = config.get("base_url")
            if base_url:
                self.client = OpenAI(base_url=base_url, api_key=config["api_key"])
            else:
                self.client = OpenAI(api_key=config["api_key"])
    
    async def evaluate(self, 
                      question: str,
                      agent_response: str,
                      expected_answer: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Use LLM to evaluate Agent response"""
        
        if not self.client:
            return {
                "score": 0.0,
                "reasoning": "LLM client not configured; provide llm_config['client'] or an API key",
                "details": {"error": "OpenAI client not initialized"}
            }
        if not self.model_name:
            return {
                "score": 0.0,
                "reasoning": "Model name missing in llm_config; please set llm_config['model']",
                "details": {"error": "LLM model not specified"}
            }
        
        try:
            evaluation_prompt = self._build_evaluation_prompt(
                question, agent_response, expected_answer, context
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a rigorous AI agent evaluation expert. "
                        "Always follow the response schema exactly."
                    ),
                },
                {"role": "user", "content": evaluation_prompt},
            ]

            sample_results = []
            for sample_index in range(self.num_samples):
                seed = self._rng.randint(0, 2**31 - 1) if self._rng else None
                sample_results.append(
                    self._evaluate_single_sample(messages, sample_index, seed)
                )

            return self._aggregate_samples(sample_results)

        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"LLM evaluation failed: {str(e)}",
                "details": {"error": str(e), "evaluation_method": "llm_based"}
            }
    
    def _build_evaluation_prompt(self, 
                                question: str, 
                                agent_response: str, 
                                expected_answer: Optional[str] = None,
                                context: Optional[Dict[str, Any]] = None) -> str:
        """Build evaluation prompt"""
        
        prompt = f"""
Please evaluate the following AI agent response quality:

Question: {question}

Agent Response: {agent_response}
"""
        
        if expected_answer:
            prompt += f"\nExpected Answer: {expected_answer}"
        
        if context:
            prompt += f"\nContext Information: {json.dumps(context, ensure_ascii=False)}"
        
        prompt += """

Please score along the following dimensions (0.0-1.0):
- accuracy: Does the response answer the question correctly?
- completeness: Does it cover key details?
- professionalism: Does it demonstrate domain knowledge?
- usefulness: Will the user benefit from it?

Return a single JSON object only (no markdown). Schema:
{
  "score": <float 0-1>,
  "confidence": <float 0-1>,
  "reasoning": "Concise justification referencing evidence",
  "flags": ["optional issues"],
  "dimensions": {
    "accuracy": <float 0-1>,
    "completeness": <float 0-1>,
    "professionalism": <float 0-1>,
    "usefulness": <float 0-1>
  }
}
If information is insufficient, set confidence <= 0.3 and explain why.
"""
        
        return prompt

    def _evaluate_single_sample(self, messages: List[Dict[str, str]], sample_index: int, seed: Optional[int]) -> Dict[str, Any]:
        """Run one LLM evaluation sample with retries for structured output."""

        last_error = ""
        last_response = ""
        for attempt in range(self.max_retries + 1):
            try:
                llm_response = self._call_llm(messages, seed=seed)
                last_response = llm_response
                parsed = self._parse_llm_response(llm_response)
                if parsed["valid"] or not self.require_valid_json:
                    if not parsed["valid"]:
                        parsed["valid"] = True
                    parsed.update(
                        {
                            "raw_response": llm_response,
                            "sample_index": sample_index,
                            "attempt": attempt + 1,
                        }
                    )
                    return parsed
                last_error = parsed["reasoning"]
            except Exception as exc:
                last_error = f"LLM invocation failed: {exc}"
            if not self.retry_invalid_json:
                break

        return {
            "score": 0.0,
            "confidence": 0.0,
            "reasoning": f"Failed to obtain valid JSON after {self.max_retries + 1} attempts. Last issue: {last_error}",
            "flags": ["invalid_llm_response"],
            "raw_response": last_response,
            "sample_index": sample_index,
            "seed": seed,
            "attempt": self.max_retries + 1,
            "valid": False,
        }

    def _call_llm(self, messages: List[Dict[str, str]], seed: Optional[int] = None) -> str:
        cache_hit = False
        cache_path = None
        cache_key = None
        if self.enable_cache:
            cache_key = self._cache_key(messages, seed)
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    if "content" in cached:
                        return cached["content"]
                except Exception:
                    pass

        request_kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if seed is not None:
            request_kwargs["seed"] = seed
        response = self.client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content

        if self.enable_cache and cache_path:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({"content": content}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        return content

    def _cache_key(self, messages: List[Dict[str, str]], seed: Optional[int]) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": seed,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(blob.encode("utf-8")).hexdigest()

    def _parse_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse and validate LLM response enforcing schema."""

        normalized = self._extract_json_blob(llm_response)
        try:
            result = json.loads(normalized)
        except json.JSONDecodeError:
            return {
                "score": 0.0,
                "confidence": 0.0,
                "reasoning": "LLM response not valid JSON; retry required.",
                "flags": ["invalid_json"],
                "dimensions": {},
                "valid": False,
            }

        fallback_reason = (llm_response or "")[:200]

        score, score_flag = self._extract_float(result, "score", 0.0, 1.0, 0.0)
        confidence, confidence_flag = self._extract_float(result, "confidence", 0.0, 1.0, 0.0)
        reasoning = self._extract_string(result, "reasoning", fallback_reason)
        flags = self._extract_list(result, "flags", default=[])
        dimensions = result.get("dimensions") if isinstance(result.get("dimensions"), dict) else {}

        schema_flags = []
        for flag in (score_flag, confidence_flag):
            if flag:
                schema_flags.append(flag)
        flags.extend(schema_flags)

        if self._contains_uncertainty(reasoning):
            confidence = min(confidence, 0.4)
            flags.append("low_confidence_reasoning")

        return {
            "score": score,
            "confidence": confidence,
            "reasoning": reasoning,
            "flags": flags,
            "dimensions": dimensions,
            "valid": self._is_valid_payload(score_flag, confidence_flag, reasoning),
        }

    def _aggregate_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid_samples = [sample for sample in samples if sample.get("valid", True)]

        dimension_scores: Dict[str, float] = {}

        if valid_samples:
            scores = [sample["score"] for sample in valid_samples]
            confidences = [sample.get("confidence", 0.0) for sample in valid_samples]
            aggregated_score = self._trimmed_mean(scores)
            aggregated_confidence = sum(confidences) / len(confidences)
            combined_reasoning = " | ".join(
                f"Sample {sample['sample_index'] + 1}: {sample['reasoning']}"
                for sample in valid_samples
            )
            flags = sorted({flag for sample in valid_samples for flag in sample.get("flags", [])})
            for sample in valid_samples:
                for dimension, value in (sample.get("dimensions") or {}).items():
                    try:
                        dimension_scores.setdefault(dimension, []).append(float(value))
                    except (TypeError, ValueError):
                        continue
        else:
            aggregated_score = 0.0
            aggregated_confidence = 0.0
            combined_reasoning = samples[0]["reasoning"] if samples else "LLM evaluation failed."
            flags = ["no_valid_llm_sample"]

        details = {
            "model_used": self.model_name,
            "confidence": aggregated_confidence,
            "flags": flags,
            "evaluation_method": "llm_based",
            "samples": samples,
            "sample_count": len(samples),
            "valid_sample_count": len(valid_samples),
            "random_seed": self.random_seed,
            "trim_ratio": self.trim_ratio,
        }

        if dimension_scores:
            details["dimension_breakdown"] = {
                name: self._trimmed_mean(values) for name, values in dimension_scores.items()
            }

        # Retain backward-compatible field for the first raw response
        if samples:
            details["llm_response"] = samples[0].get("raw_response", "")

        return {
            "score": aggregated_score,
            "reasoning": combined_reasoning,
            "details": details,
        }

    def _trimmed_mean(self, values: List[float]) -> float:
        if not values:
            return 0.0
        if self.trim_ratio <= 0 or len(values) < 3:
            return sum(values) / len(values)
        sorted_values = sorted(values)
        trim = min(len(sorted_values) // 2, max(0, int(len(sorted_values) * self.trim_ratio)))
        if trim == 0:
            return sum(sorted_values) / len(sorted_values)
        trimmed = sorted_values[trim:-trim] if trim > 0 else sorted_values
        if not trimmed:
            return sum(sorted_values) / len(sorted_values)
        return sum(trimmed) / len(trimmed)

    def _extract_json_blob(self, text: str) -> str:
        if not text:
            return ""
        snippet = text.strip()
        if snippet.startswith("```"):
            snippet = re.sub(r"^```[a-zA-Z0-9]*", "", snippet).strip("` \n")
        start = snippet.find("{")
        end = snippet.rfind("}")
        if start != -1 and end != -1 and end > start:
            return snippet[start : end + 1]
        return snippet

    def _is_valid_payload(self, score_flag: Optional[str], confidence_flag: Optional[str], reasoning: str) -> bool:
        if not self.enforce_schema:
            return True
        return score_flag is None and confidence_flag is None and bool(reasoning.strip())

    def _extract_float(self, payload: Dict[str, Any], key: str, min_value: float, max_value: float, default: float) -> Tuple[float, Optional[str]]:
        value = payload.get(key)
        try:
            num = float(value)
            if not (min_value <= num <= max_value):
                return default, f"{key}_out_of_range"
            return num, None
        except (TypeError, ValueError):
            return default, f"{key}_invalid"

    def _extract_string(self, payload: Dict[str, Any], key: str, default: str) -> str:
        value = payload.get(key)
        return str(value) if isinstance(value, str) and value.strip() else default

    def _extract_list(self, payload: Dict[str, Any], key: str, default: Optional[List[str]] = None) -> List[str]:
        value = payload.get(key)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return value
        return default or []

    def _contains_uncertainty(self, text: str) -> bool:
        lowered = (text or "").lower()
        uncertainty_tokens = [
            "unsure",
            "uncertain",
            "cannot determine",
            "not enough information",
            "insufficient data",
            "no confidence",
        ]
        return any(token in lowered for token in uncertainty_tokens)
    
    def get_capabilities(self) -> List[str]:
        """Get evaluator capabilities."""
        return [
            "general_evaluation",
            "factual_accuracy",
            "completeness",
            "reasoning_quality",
            "tool_use_evaluation",
            "domain_knowledge",
        ]
