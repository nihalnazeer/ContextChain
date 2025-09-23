"""
Adaptive Context Budgeting Algorithm (ACBA) v2.1
Integrates:
- Hierarchical Budget Policy Optimization (Lyu et al.)
- Multi-Armed Bandits (Bouneffouf et al.)
- RL-driven Compression (Cui et al.)

Alignment:
- Compatible with core.py (compute_optimal_budget(query, documents, context), update_with_feedback(budget, performance_metrics, context))
- Exposes complexity_assessor attribute for direct access. Note: The assess_complexity method is part of the QueryComplexityAssessor class, accessible via self.complexity_assessor.assess_complexity(query), not directly on AdaptiveContextBudgetingAlgorithm.
- BudgetAllocation fields match llm.py expectations (generation_tokens, retrieval_tokens, compression_tokens)
- Fixes contextual boost indexing bug; adds decaying exploration and richer rewards
"""
import re
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict # <--- FIX: Imported asdict
from enum import Enum
import asyncio
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Optional semantic features (enabled if available)
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    ST_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    st_util = None
    ST_AVAILABLE = False

# -------------------------
# Bandit & Allocation Types
# -------------------------

class BudgetArm(Enum):
    """Budget allocation strategies as bandit arms"""
    LIGHT_RETRIEVE = 0       # Low k, minimal compression
    HEAVY_RETRIEVE = 1       # High k, heavy compression
    SPARSE_COMPRESS = 2      # Medium k, extractive compression
    DENSE_COMPRESS = 3       # Medium k, abstractive compression
    ADAPTIVE_COMPRESS = 4    # Query-aware compression
    HYBRID_OPTIMIZE = 5      # Multi-vector + rerank

@dataclass
class BudgetAllocation:
    """Hierarchical budget allocation record"""
    retrieval_tokens: int
    compression_tokens: int
    generation_tokens: int
    total_budget: int
    arm_selected: BudgetArm
    confidence_score: float
    hierarchy_weights: Dict[str, float]
    expected_utility: float
    allocation_timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval_tokens": self.retrieval_tokens,
            "compression_tokens": self.compression_tokens,
            "generation_tokens": self.generation_tokens,
            "total_budget": self.total_budget,
            "arm_selected": self.arm_selected.name,
            "confidence_score": self.confidence_score,
            "hierarchy_weights": self.hierarchy_weights,
            "expected_utility": self.expected_utility,
            "allocation_timestamp": self.allocation_timestamp.isoformat(),
        }

@dataclass
class QueryComplexity:
    """Query complexity assessment for budget optimization"""
    semantic_complexity: float       # 0-1
    compositional_complexity: float  # 0-1
    temporal_complexity: float       # 0-1
    domain_complexity: float         # 0-1
    overall_score: float

    # --- FIX: Added a .to_dict() method using asdict ---
    def to_dict(self) -> Dict[str, float]:
        """Converts the dataclass instance to a dictionary."""
        return asdict(self)

# -------------------------
# Thompson Sampling Bandit
# -------------------------

class ThompsonSamplingBandit:
    """
    Thompson Sampling Multi-Armed Bandit (non-stationary, contextual boosts)
    Fixes: proper integer indexing for arms; optional decaying exploration epsilon
    """

    def __init__(self, n_arms: int = 6, alpha: float = 1.0, beta: float = 1.0, initial_epsilon: float = 0.2, epsilon_decay: float = 0.999):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms) * alpha
        self.beta = np.ones(n_arms) * beta
        self.arm_counts = np.zeros(n_arms)
        self.contextual_features: Dict[str, np.ndarray] = {}
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay

    @staticmethod
    def arm_to_index(arm: Union[BudgetArm, int]) -> int:
        return arm if isinstance(arm, int) else int(arm.value)

    def select_arm(self, context: Optional[Dict] = None) -> int:
        # Epsilon-greedy exploration on top of TS to avoid premature convergence
        if np.random.rand() < self.epsilon:
            choice = np.random.randint(self.n_arms)
            self._decay_epsilon()
            return choice

        samples = np.random.beta(self.alpha, self.beta)

        # Contextual boost uses integer indices
        if context:
            contextual_boost = self._compute_contextual_boost(context)
            samples = samples + contextual_boost

        self._decay_epsilon()
        return int(np.argmax(samples))

    def update(self, arm: Union[int, BudgetArm], reward: float, context: Optional[Dict] = None):
        idx = self.arm_to_index(arm)
        # Bounded reward update
        reward = float(np.clip(reward, 0.0, 1.0))
        if reward >= 0.5:
            self.alpha[idx] += reward
        else:
            self.beta[idx] += (1.0 - reward)
        self.arm_counts[idx] += 1.0

        if context:
            self._update_contextual_features(idx, context, reward)

    def _compute_contextual_boost(self, context: Dict) -> np.ndarray:
        boost = np.zeros(self.n_arms, dtype=float)
        complexity = float(context.get("complexity", 0.5))
        # Complex → favor heavy retrieval and adaptive compression
        if complexity > 0.7:
            boost[BudgetArm.HEAVY_RETRIEVE.value] += 0.10
            boost[BudgetArm.ADAPTIVE_COMPRESS.value] += 0.10
        # Simple → favor light retrieve
        elif complexity < 0.3:
            boost[BudgetArm.LIGHT_RETRIEVE.value] += 0.10
        return boost

    def _update_contextual_features(self, arm_idx: int, context: Dict, reward: float):
        # Track EMA rewards by query_type
        qt = context.get("query_type", "general")
        if qt not in self.contextual_features:
            self.contextual_features[qt] = np.zeros(self.n_arms, dtype=float)
        ema = 0.1
        self.contextual_features[qt][arm_idx] = (1 - ema) * self.contextual_features[qt][arm_idx] + ema * reward

    def _decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

# -----------------------------------------
# Hierarchical Budget Policy (Lyu et al.)
# -----------------------------------------

class HierarchicalBudgetOptimizer:
    """
    Hierarchical Budget Policy Optimization
    """

    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        # Base importance weights (critical=generation, important=retrieval, support=compression)
        self.hierarchy_levels = {"critical": 0.5, "important": 0.3, "support": 0.2}
        self.learned_weights = self._initialize_learned_weights()

    def compute_hierarchical_allocation(self, complexity: QueryComplexity, arm: BudgetArm) -> Dict[str, int]:
        adapted = self._adapt_hierarchy_to_complexity(complexity)
        strat = self._get_arm_allocation_strategy(arm)
        final_w = self._combine_weights(adapted, strat)
        alloc = {
            "generation": int(self.max_tokens * final_w["generation"]),
            "retrieval": int(self.max_tokens * final_w["retrieval"]),
            "compression": int(self.max_tokens * final_w["compression"]),
        }
        return self._normalize_allocation(alloc)

    def _adapt_hierarchy_to_complexity(self, complexity: QueryComplexity) -> Dict[str, float]:
        w = self.hierarchy_levels.copy()
        if complexity.overall_score > 0.7:
            w["important"] += 0.10  # retrieval
            w["critical"] -= 0.05   # generation
            w["support"] -= 0.05    # compression
        elif complexity.overall_score < 0.3:
            w["critical"] += 0.10
            w["important"] -= 0.10
        return {"generation": w["critical"], "retrieval": w["important"], "compression": w["support"]}

    def _get_arm_allocation_strategy(self, arm: BudgetArm) -> Dict[str, float]:
        strategies = {
            BudgetArm.LIGHT_RETRIEVE:  {"generation": 0.60, "retrieval": 0.20, "compression": 0.20},
            BudgetArm.HEAVY_RETRIEVE:  {"generation": 0.40, "retrieval": 0.40, "compression": 0.20},
            BudgetArm.SPARSE_COMPRESS: {"generation": 0.50, "retrieval": 0.30, "compression": 0.20},
            BudgetArm.DENSE_COMPRESS:  {"generation": 0.40, "retrieval": 0.30, "compression": 0.30},
            BudgetArm.ADAPTIVE_COMPRESS: {"generation": 0.45, "retrieval": 0.35, "compression": 0.20},
            BudgetArm.HYBRID_OPTIMIZE: {"generation": 0.40, "retrieval": 0.35, "compression": 0.25},
        }
        return strategies.get(arm, strategies[BudgetArm.ADAPTIVE_COMPRESS])

    def _combine_weights(self, hierarchy: Dict[str, float], strategy: Dict[str, float]) -> Dict[str, float]:
        alpha, beta = 0.7, 0.3
        return {k: alpha * hierarchy[k] + beta * strategy[k] for k in hierarchy.keys()}

    def _normalize_allocation(self, allocation: Dict[str, int]) -> Dict[str, int]:
        total = sum(allocation.values())
        if total > self.max_tokens:
            scale = self.max_tokens / max(total, 1)
            allocation = {k: int(v * scale) for k, v in allocation.items()}
        allocation["generation"] = max(allocation["generation"], 100)
        allocation["retrieval"] = max(allocation["retrieval"], 50)
        allocation["compression"] = max(allocation["compression"], 30)
        return allocation

    def _initialize_learned_weights(self) -> Dict[str, Any]:
        return {"complexity_sensitivity": 1.0, "domain_adaptation": {}, "temporal_weights": {}}

# -----------------------------------------
# RL-driven Compression (Cui et al., CORE)
# -----------------------------------------

class RLCompressionAgent:
    """
    RL-based Compression Agent
    - Generates multiple candidates (extractive, abstractive, hybrid, query-focused)
    - Scores candidates with reward model + optional semantic features
    - Falls back to extractive if threshold not met
    """

    def __init__(self, model_path: Optional[str] = None, enable_semantic: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compression_model = self._initialize_model()  # placeholder NN
        self.reward_model = self._initialize_reward_model()
        self.optimization_history: List[Dict[str, Any]] = []
        self.enable_semantic = enable_semantic and ST_AVAILABLE
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2") if self.enable_semantic else None

    def _initialize_model(self) -> nn.Module:
        class CompressionTransformer(nn.Module):
            def __init__(self, d_model=256, nhead=8, num_layers=3):  # lighter defaults
                super().__init__()
                self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
                self.embedding = nn.Embedding(50000, d_model)
                self.output_projection = nn.Linear(d_model, 50000)

            def forward(self, src, tgt):
                src_emb = self.embedding(src)
                tgt_emb = self.embedding(tgt)
                out = self.transformer(src_emb, tgt_emb)
                return self.output_projection(out)
        return CompressionTransformer().to(self.device)

    def _initialize_reward_model(self) -> nn.Module:
        class RewardModel(nn.Module):
            def __init__(self, input_dim=1024):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid(),
                )
            def forward(self, x): return self.net(x)
        return RewardModel().to(self.device)

    async def compress_with_rl_optimization(self, docs: List[str], query: str, target_length: int, quality_threshold: float = 0.8) -> Tuple[str, float]:
        candidates = await self._generate_compression_candidates(docs, query, target_length)
        scored: List[Tuple[str, float]] = []
        for cand in candidates:
            feats = self._extract_compression_features(cand, docs, query)
            reward = float(self.reward_model(feats).item())
            scored.append((cand, reward))
        valid = [(c, s) for c, s in scored if s >= quality_threshold]
        if not valid:
            logger.warning("No candidates met quality threshold; using extractive fallback")
            return self._extractive_fallback(docs, target_length), quality_threshold
        best_cand, best_score = max(valid, key=lambda x: x[1])
        self._store_compression_example(docs, query, best_cand, best_score)
        return best_cand, best_score

    async def _generate_compression_candidates(self, docs: List[str], query: str, target_length: int) -> List[str]:
        cands = []
        cands.append(self._extractive_compression(docs, target_length))
        cands.append(await self._abstractive_compression(docs, query, target_length))
        cands.append(self._hybrid_compression(docs, query, target_length))
        cands.append(self._query_focused_compression(docs, query, target_length))
        return cands

    def _extractive_compression(self, docs: List[str], target_length: int) -> str:
        sents: List[str] = []
        for d in docs:
            sents.extend([s.strip() for s in d.split(".") if s.strip()])
        if len(sents) <= 1: return " ".join(docs)[:target_length]

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vect = TfidfVectorizer(stop_words="english")
        tfidf = vect.fit_transform(sents)
        centrality = []
        for i in range(len(sents)):
            sim = cosine_similarity(tfidf[i], tfidf).flatten()
            centrality.append((float(np.mean(sim)), sents[i]))
        compressed, acc = "", 0
        for score, sent in sorted(centrality, reverse=True):
            if len(compressed) + len(sent) + 2 <= target_length:
                compressed += sent + ". "
                acc += 1
            if len(compressed) >= target_length: break
        return compressed.strip()

    async def _abstractive_compression(self, docs: List[str], query: str, target_length: int) -> str:
        text = " ".join(docs)
        sents = [s.strip() for s in text.split(".") if s.strip()]
        if len(sents) <= 2: return text[:target_length]
        compressed = []
        remain = target_length
        for s in sents:
            if len(s) + 2 < remain:
                compressed.append(s)
                remain -= len(s) + 2
            if remain < 50: break
        return ". ".join(compressed).strip()

    def _hybrid_compression(self, docs: List[str], query: str, target_length: int) -> str:
        ext = self._extractive_compression(docs, int(target_length * 0.7))
        remain = target_length - len(ext)
        if remain > 50:
            return (ext + " Summary: " + " ".join(docs)[:remain]).strip()
        return ext

    def _query_focused_compression(self, docs: List[str], query: str, target_length: int) -> str:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        if not docs: return ""
        all_text = docs + [query]
        vect = TfidfVectorizer(stop_words="english")
        tfidf = vect.fit_transform(all_text)
        q_vec = tfidf[-1]
        doc_vecs = tfidf[:-1]
        sims = cosine_similarity(q_vec, doc_vecs).flatten()
        weighted_sents: List[Tuple[float, str]] = []
        for d, sim in zip(docs, sims):
            for s in [x.strip() for x in d.split(".") if x.strip()]:
                weighted_sents.append((float(sim), s))
        compressed = ""
        for sim, s in sorted(weighted_sents, key=lambda x: x[0], reverse=True):
            if len(compressed) + len(s) + 2 <= target_length:
                compressed += s + ". "
        return compressed.strip()

    def _extractive_fallback(self, docs: List[str], target_length: int) -> str:
        return self._extractive_compression(docs, target_length)

    def _extract_compression_features(self, compressed: str, original_docs: List[str], query: str) -> torch.Tensor:
        feats: List[float] = []
        # 1) Compression ratio
        original_len = sum(len(d) for d in original_docs)
        ratio = (len(compressed) / max(original_len, 1)) if original_len > 0 else 0.0
        feats.append(ratio)
        # 2) Query overlap
        q_words = set(query.lower().split())
        c_words = set(compressed.lower().split())
        overlap = (len(q_words & c_words) / max(len(q_words), 1)) if q_words else 0.0
        feats.append(overlap)
        # 3) Information density
        c_list = compressed.lower().split()
        density = (len(set(c_list)) / max(len(c_list), 1)) if c_list else 0.0
        feats.append(density)
        # 4) Sentence completeness
        sents = [s for s in compressed.split(".") if s.strip()]
        completeness = (len(sents) / max(len(compressed.split(".")), 1)) if compressed else 0.0
        feats.append(completeness)
        # 5) Semantic similarity to originals (optional)
        if self.enable_semantic and self.embedding_model and ST_AVAILABLE and original_docs:
            try:
                doc_emb = self.embedding_model.encode(" ".join(original_docs), convert_to_tensor=True, normalize_embeddings=True)
                comp_emb = self.embedding_model.encode(compressed, convert_to_tensor=True, normalize_embeddings=True)
                sim = float(st_util.cos_sim(doc_emb, comp_emb).item())
                feats.append(sim)
            except Exception:
                feats.append(0.0)
        else:
            feats.append(0.0)
        # Pad to 1024
        if len(feats) < 1024:
            feats.extend([0.0] * (1024 - len(feats)))
        return torch.tensor(feats, dtype=torch.float32, device=self.device)

    def _store_compression_example(self, docs: List[str], query: str, compressed: str, score: float):
        self.optimization_history.append({
            "original_docs": docs, "query": query, "compressed": compressed,
            "reward": float(score), "timestamp": datetime.utcnow()
        })
        if len(self.optimization_history) > 10000:
            self.optimization_history = self.optimization_history[-10000:]

# -----------------------------------------
# Query Complexity Assessor
# -----------------------------------------

class QueryComplexityAssessor:
    """Assess query complexity for adaptive budget allocation"""

    def __init__(self):
        self.complexity_indicators = {
            "multi_hop": ["and then", "after that", "because of", "as a result"],
            "temporal": ["when", "before", "after", "during", "since", "until", "timeline", "history"],
            "comparative": ["compare", "versus", "vs", "better than", "worse than", "difference"],
            "causal": ["why", "because", "due to", "caused by", "reason for", "cause"],
            "quantitative": ["how many", "how much", "percentage", "ratio", "statistics", "count", "number"],
        }

    def assess_complexity(self, query: str) -> QueryComplexity:
        q = (query or "").lower()
        semantic = min(len(q.split()) / 20.0, 1.0)
        comp = 0.0
        for k, indicators in self.complexity_indicators.items():
            if k == "temporal":
                continue
            matches = sum(1 for ind in indicators if re.search(r'\b{}\b'.format(re.escape(ind)), q))
            if matches > 0:
                comp += min(matches / 3.0, 0.3)

        comp = min(comp, 1.0)
        temporal_matches = sum(1 for ind in self.complexity_indicators["temporal"] if ind in q)
        temporal = min(temporal_matches / 3.0, 1.0)
        specialized_terms = ["algorithm", "protocol", "methodology", "analysis", "framework", "optimization", "architecture"]
        domain = min(sum(1 for t in specialized_terms if t in q) / 5.0, 1.0)
        overall = (0.3 * semantic + 0.4 * comp + 0.2 * temporal + 0.1 * domain)
        return QueryComplexity(semantic, comp, temporal, domain, overall)

# -----------------------------------------
# ACBA Main
# -----------------------------------------

class AdaptiveContextBudgetingAlgorithm:
    """
    Joint optimization of retrieval, compression, and generation budgets
    Compatible with ContextChain core orchestrator and LLM optimizer
    Note: The assess_complexity method is part of the QueryComplexityAssessor class, accessible via self.complexity_assessor.assess_complexity(query), not directly on this class.
    """

    def __init__(self, max_tokens: int = 4096, learning_rate: float = 0.01, exploration_rate: float = 0.1):
        self.max_tokens = max_tokens
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate

        # Core components
        self.bandit = ThompsonSamplingBandit(n_arms=len(BudgetArm), initial_epsilon=exploration_rate)
        self.hierarchical_optimizer = HierarchicalBudgetOptimizer(max_tokens)
        self.rl_compressor = RLCompressionAgent()
        # IMPORTANT: expose complexity_assessor attribute for core.py
        self.complexity_assessor = QueryComplexityAssessor()

        # Learning & monitoring
        self.allocation_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[datetime, Dict[str, Any]] = {}
        self.adaptation_count = 0

        logger.info(f"ACBA initialized with max_tokens={max_tokens}")

    # Core.py compatible signature
    async def compute_optimal_budget(self, query: str, documents: List[Dict], context: Optional[Dict] = None) -> BudgetAllocation:
        start_time = datetime.utcnow()
        complexity = self.complexity_assessor.assess_complexity(query)

        bandit_context = {
            "query_type": self._classify_query_type(query),
            "complexity": complexity.overall_score,
            "doc_count": len(documents or []),
            "avg_doc_length": float(np.mean([len(d.get("content", "")) for d in (documents or [])])) if documents else 0.0,
        }

        arm_idx = self.bandit.select_arm(bandit_context)
        selected_arm = list(BudgetArm)[arm_idx]

        token_alloc = self.hierarchical_optimizer.compute_hierarchical_allocation(complexity, selected_arm)
        expected_utility = self._predict_utility(query, complexity, selected_arm, token_alloc)

        allocation = BudgetAllocation(
            retrieval_tokens=token_alloc["retrieval"],
            compression_tokens=token_alloc["compression"],
            generation_tokens=token_alloc["generation"],
            total_budget=sum(token_alloc.values()),
            arm_selected=selected_arm,
            confidence_score=self._compute_confidence(complexity, selected_arm),
            hierarchy_weights=self.hierarchical_optimizer.learned_weights,
            expected_utility=expected_utility,
            allocation_timestamp=start_time,
        )

        self._store_allocation_decision(allocation, bandit_context)
        logger.info(f"ACBA allocated: arm={allocation.arm_selected.name}, gen={allocation.generation_tokens}, ret={allocation.retrieval_tokens}, comp={allocation.compression_tokens}, EU={expected_utility:.3f}")
        return allocation

    # Core.py compatible signature
    async def update_with_feedback(self, budget: BudgetAllocation, performance_metrics: Dict[str, float], context: Optional[Dict] = None):
        allocation = budget
        actual = performance_metrics

        reward = self._compute_reward_signal(allocation, actual)

        bandit_context = {
            "query_type": (context or {}).get("query_type", "general"),
            "complexity": (context or {}).get("complexity", 0.5),
            "session_id": (context or {}).get("session_id"),
        }

        self.bandit.update(allocation.arm_selected, reward, bandit_context)
        self._update_hierarchical_weights(allocation, actual, reward)

        self.performance_metrics[allocation.allocation_timestamp] = {
            "allocation": allocation,
            "performance": actual,
            "reward": reward,
        }
        self.adaptation_count += 1
        logger.info(f"ACBA feedback: reward={reward:.3f}, adaptations={self.adaptation_count}")

    # -----------------
    # Helper functions
    # -----------------

    def _classify_query_type(self, query: str) -> str:
        q = (query or "").lower()
        if any(w in q for w in ["analyze", "analysis", "trend", "pattern"]): return "analytical"
        if any(w in q for w in ["compare", "versus", "vs", "difference"]):   return "comparative"
        if any(w in q for w in ["when", "timeline", "history", "chronology"]): return "temporal"
        if any(w in q for w in ["why", "because", "reason", "cause"]):       return "causal"
        if any(w in q for w in ["how many", "count", "number", "quantity"]): return "quantitative"
        return "general"

    def _predict_utility(self, query: str, complexity: QueryComplexity, arm: BudgetArm, allocation: Dict[str, int]) -> float:
        base = 0.5
        bonus = 0.0
        if complexity.overall_score > 0.7 and arm in [BudgetArm.HEAVY_RETRIEVE, BudgetArm.ADAPTIVE_COMPRESS]:
            bonus += 0.20
        elif complexity.overall_score < 0.3 and arm == BudgetArm.LIGHT_RETRIEVE:
            bonus += 0.15
        total_budget = sum(allocation.values())
        bonus += 0.10 if total_budget <= self.max_tokens * 0.9 else 0.0
        hist = self._get_historical_performance_bonus(arm)
        return float(min(base + bonus + hist, 1.0))

    def _compute_confidence(self, complexity: QueryComplexity, arm: BudgetArm) -> float:
        idx = int(arm.value)
        base = min(self.bandit.arm_counts[idx] / 100.0, 0.8)
        if complexity.overall_score > 0.7 and arm in [BudgetArm.HEAVY_RETRIEVE, BudgetArm.ADAPTIVE_COMPRESS]:
            base += 0.15
        elif complexity.overall_score < 0.3 and arm == BudgetArm.LIGHT_RETRIEVE:
            base += 0.10
        return float(min(base, 1.0))

    def _store_allocation_decision(self, allocation: BudgetAllocation, context: Dict):
        rec = {"timestamp": allocation.allocation_timestamp, "allocation": allocation, "context": context, "adaptation_count": self.adaptation_count}
        self.allocation_history.append(rec)
        if len(self.allocation_history) > 5000:
            self.allocation_history = self.allocation_history[-5000:]

    def _compute_reward_signal(self, allocation: BudgetAllocation, performance: Dict[str, float]) -> float:
        # Inputs may be present as 'quality' and 'latency' in ms from core.py
        task_perf = float(performance.get("accuracy", performance.get("quality", 0.0)))
        tokens_used = float(performance.get("tokens_used", allocation.total_budget))
        efficiency = (task_perf / (tokens_used / 1000.0)) if tokens_used > 0 else 0.0
        latency_ms = performance.get("latency", performance.get("latency_seconds", 1000.0))
        latency_sec = latency_ms / 1000.0 if latency_ms > 10 else float(latency_ms)
        latency_penalty = max(0.0, (latency_sec - 2.0) * 0.1)
        budget_adherence = 1.0 - abs(tokens_used - allocation.total_budget) / max(allocation.total_budget, 1.0)
        budget_bonus = max(0.0, budget_adherence) * 0.1
        # Optional external signals
        hf = float(performance.get("human_feedback", 0.0))  # 0-1
        judge = float(performance.get("llm_judge_score", 0.0))  # 0-1
        extra = 0.1 * hf + 0.1 * judge
        total = 0.6 * task_perf + 0.2 * min(efficiency, 1.0) + 0.1 * budget_bonus - 0.1 * latency_penalty + extra
        return float(np.clip(total, 0.0, 1.0))

    def _update_hierarchical_weights(self, allocation: BudgetAllocation, performance: Dict[str, float], reward: float):
        lr = self.learning_rate
        err = reward - allocation.expected_utility
        self.hierarchical_optimizer.learned_weights["complexity_sensitivity"] += lr * err
        self.hierarchical_optimizer.learned_weights["complexity_sensitivity"] = float(np.clip(self.hierarchical_optimizer.learned_weights["complexity_sensitivity"], 0.1, 2.0))

    def _get_historical_performance_bonus(self, arm: BudgetArm) -> float:
        if not self.performance_metrics: return 0.0
        recent_threshold = datetime.utcnow() - timedelta(hours=24)
        rewards = []
        for ts, m in self.performance_metrics.items():
            if ts > recent_threshold and m["allocation"].arm_selected == arm:
                rewards.append(float(m["reward"]))
        return float(((np.mean(rewards) - 0.5) * 0.2)) if rewards else 0.0

    def get_performance_summary(self) -> Dict[str, Any]:
        if not self.performance_metrics:
            return {"status": "no_data"}
        recent_threshold = datetime.utcnow() - timedelta(hours=24)
        recent = [m for ts, m in self.performance_metrics.items() if ts > recent_threshold]
        if not recent:
            return {"status": "no_recent_data"}
        rewards = [m["reward"] for m in recent]
        arm_dist: Dict[str, int] = {}
        for m in recent:
            arm = m["allocation"].arm_selected.name
            arm_dist[arm] = arm_dist.get(arm, 0) + 1
        return {
            "status": "active",
            "total_adaptations": self.adaptation_count,
            "recent_queries": len(recent),
            "avg_reward_24h": float(np.mean(rewards)),
            "reward_std_24h": float(np.std(rewards)),
            "arm_distribution": arm_dist,
            "bandit_arm_counts": self.bandit.arm_counts.tolist(),
            "learned_complexity_sensitivity": float(self.hierarchical_optimizer.learned_weights.get("complexity_sensitivity", 1.0)),
            "success_rate": (len([r for r in rewards if r > 0.5]) / max(len(rewards), 1)),
            "avg_tokens_saved": float(np.mean([
                m["allocation"].total_budget - m["performance"].get("tokens_used", m["allocation"].total_budget) for m in recent
            ])) if recent else 0.0,
        }

# ---------------------------
# Minimal test harness (dev)
# ---------------------------

async def _simulate_queries(acba: AdaptiveContextBudgetingAlgorithm, n: int = 20) -> Dict[str, Any]:
    """
    Dev harness: simulate queries, allocate budgets, and update with synthetic rewards.
    """
    import random
    rng = np.random.default_rng(42)
    types = ["analytical", "comparative", "temporal", "causal", "quantitative", "general"]
    for i in range(n):
        qt = random.choice(types)
        query = f"Simulated {qt} query number {i}"
        docs = [{"content": "Lorem ipsum dolor sit amet. " * rng.integers(5, 20)} for _ in range(rng.integers(0, 6))]
        alloc = await acba.compute_optimal_budget(query, docs, context={"query_type": qt})
        # Fake performance: higher reward for matched arms to complexity
        perf_quality = rng.uniform(0.6, 0.95)
        qc = acba.complexity_assessor.assess_complexity(query)
        if qc.overall_score > 0.6 and alloc.arm_selected in [BudgetArm.HEAVY_RETRIEVE, BudgetArm.ADAPTIVE_COMPRESS]:
            perf_quality = min(1.0, perf_quality + 0.15)
        elif qc.overall_score < 0.4 and alloc.arm_selected == BudgetArm.LIGHT_RETRIEVE:
            perf_quality = min(1.0, perf_quality + 0.1)
        
        # Simulate latency and token usage
        latency_ms = rng.uniform(500, 3000)
        tokens_used = int(alloc.total_budget * rng.uniform(0.8, 1.1))

        perf_metrics = {"quality": perf_quality, "latency": latency_ms, "tokens_used": tokens_used}
        await acba.update_with_feedback(alloc, perf_metrics, context={"query_type": qt, "complexity": qc.overall_score})
        await asyncio.sleep(0.01)

    summary = acba.get_performance_summary()
    logger.info("=" * 20 + " SIMULATION SUMMARY " + "=" * 20)
    for k, v in summary.items():
        logger.info(f"{k}: {v}")
    return summary


if __name__ == "__main__":
    import asyncio
    async def main():
        acba = AdaptiveContextBudgetingAlgorithm(max_tokens=2048, learning_rate=0.02, exploration_rate=0.15)
        summary = await _simulate_queries(acba, n=30)
        print("Simulation summary:", summary)
    asyncio.run(main())