"""Detection mode enumerations for simplified configuration."""
from dataclasses import dataclass
from enum import Enum

# Legacy pipeline imports removed
from services.aggressive_similarity_pipeline import AggressivePipelineConfig


class DetectionMode(str, Enum):
    """Pre-configured detection modes to simplify configuration."""

    PURE_SEMANTIC = "pure_semantic"  # Semantic only, no lexical
    AGGRESSIVE = "aggressive"  # Semantic + MinHash + Cross-encoder
    FAST = "fast"  # Semantic + Light filtering
    STRICT = "strict"  # All filters + High thresholds


@dataclass
class DetectionModeConfig:
    """Configuration for a detection mode."""
    mode: DetectionMode
    name: str
    description: str
    # All modes use aggressive pipeline
    aggressive_config: AggressivePipelineConfig | None = None


# Pre-configured detection modes
DETECTION_MODE_CONFIGS = {
    DetectionMode.PURE_SEMANTIC: DetectionModeConfig(
        mode=DetectionMode.PURE_SEMANTIC,
        name="Pure Semantic",
        description="Semantic embeddings only, no lexical analysis",
        aggressive_config=AggressivePipelineConfig(
            semantic_threshold=0.70,
            final_threshold=0.30,  # Drastically lowered for Qwen3-Reranker-0.6B (Chinese text scores ~0.3-0.5)
            top_k=20,
            use_minhash=False,
        )
    ),

    DetectionMode.AGGRESSIVE: DetectionModeConfig(
        mode=DetectionMode.AGGRESSIVE,
        name="Aggressive Detection",
        description="Semantic + MinHash + Cross-encoder for maximum detection",
        aggressive_config=AggressivePipelineConfig(
            semantic_threshold=0.65,
            final_threshold=0.15,  # Drastically lowered for Qwen3-Reranker-0.6B (Chinese text scores ~0.1-0.5)
            top_k=50,
            use_minhash=True,
            minhash_threshold=0.3,
            cross_encoder_threshold=0.50,
        )
    ),

    DetectionMode.FAST: DetectionModeConfig(
        mode=DetectionMode.FAST,
        name="Fast Detection",
        description="Semantic with light filtering for speed",
        aggressive_config=AggressivePipelineConfig(
            semantic_threshold=0.75,
            final_threshold=0.40,  # Drastically lowered for Qwen3-Reranker-0.6B (Chinese text scores ~0.3-0.5)
            top_k=10,
            use_minhash=False,
        )
    ),

    DetectionMode.STRICT: DetectionModeConfig(
        mode=DetectionMode.STRICT,
        name="Strict Detection",
        description="All filters with high thresholds for precision",
        aggressive_config=AggressivePipelineConfig(
            semantic_threshold=0.80,
            final_threshold=0.50,  # Lowered for Qwen3-Reranker-0.6B (only identical Chinese text reaches ~0.5)
            top_k=30,
            use_minhash=True,
            minhash_threshold=0.5,
            cross_encoder_threshold=0.70,
        )
    ),
}


def get_detection_config(
    mode: DetectionMode,
    semantic_threshold_override: float | None = None,
    final_threshold_override: float | None = None,
    top_k_override: int | None = None,
) -> DetectionModeConfig:
    """Get detection configuration with optional overrides."""

    config = DETECTION_MODE_CONFIGS[mode]

    # Apply overrides if provided
    if config.aggressive_config:
        # Clone the config to avoid modifying the original
        from copy import deepcopy
        config = deepcopy(config)

        # After deepcopy, aggressive_config should still exist
        if config.aggressive_config:
            if semantic_threshold_override is not None:
                config.aggressive_config.semantic_threshold = semantic_threshold_override
            if final_threshold_override is not None:
                config.aggressive_config.final_threshold = final_threshold_override
            if top_k_override is not None:
                config.aggressive_config.top_k = top_k_override

    return config
