"""
Context Engineer v2.0
Advanced prompt construction with embedding-informed optimization
Integrates: Huang et al. (2025) embedding-informed retrieval patterns
"""
from __future__ import annotations

import re
import json
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging
from pathlib import Path
import jinja2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .acba import BudgetAllocation, QueryComplexity
from .dag import ExecutionContext  # Import for type hinting

logger = logging.getLogger(__name__)

class PromptStyle(Enum):
    """Available prompt construction styles"""
    QA_STRUCTURED = "qa_structured"
    REASONING_FRIENDLY = "reasoning_friendly"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"
    COMPLIANCE = "compliance"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"

class DocumentOrder(Enum):
    """Document ordering strategies"""
    RELEVANCE_SCORE = "relevance_score"
    COMPRESSION_RETENTION = "compression_retention"
    TEMPORAL = "temporal"
    AUTHORITY = "authority"
    HYBRID_WEIGHTED = "hybrid_weighted"

@dataclass
class DocumentMetadata:
    """Enhanced document metadata for provenance tracking"""
    source_id: str
    url: Optional[str] = None
    retrieval_score: float = 0.0
    vector_score: float = 0.0
    compression_ratio: float = 1.0
    compression_loss: float = 0.0
    last_updated: Optional[datetime] = None
    authority_score: float = 0.5
    temporal_relevance: float = 0.5
    token_count: int = 0

@dataclass
class ProcessedDocument:
    """Document with processing metadata"""
    content: str
    metadata: DocumentMetadata
    processing_info: Dict[str, Any]

@dataclass
class PromptConstructionConfig:
    """Configuration for prompt construction"""
    style: PromptStyle = PromptStyle.QA_STRUCTURED
    max_prompt_tokens: int = 2048
    include_metadata: bool = True
    include_provenance: bool = True
    include_citations: bool = True
    order_strategy: DocumentOrder = DocumentOrder.HYBRID_WEIGHTED
    safety_filtering: bool = True
    compliance_mode: bool = False
    few_shot_examples: int = 3
    cot_enabled: bool = True
    quality_threshold: float = 0.3

class EmbeddingInformedOptimizer:
    """
    Embedding-informed document optimization following Huang et al. (2025)
    Ensures high-quality, relevant context for LLM generation
    """
    
    def __init__(self, quality_threshold: float = 0.7):
        self.quality_threshold = quality_threshold
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.quality_cache = {}
        
    def assess_document_quality(self, doc: ProcessedDocument, query: str) -> float:
        """
        Assess document quality for inclusion following Huang et al. methodology
        """
        cache_key = f"{hash(doc.content)}_{hash(query)}"
        if cache_key in self.quality_cache:
            return self.quality_cache[cache_key]
        
        # Quality factors based on Huang et al. embedding-informed approach
        try:
            docs_and_query = [doc.content, query]
            tfidf_matrix = self.vectorizer.fit_transform(docs_and_query)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            semantic_score = float(similarity)
        except:
            semantic_score = 0.5  # Fallback
        
        # Information density
        words = doc.content.split()
        unique_words = set(words)
        density_score = len(unique_words) / len(words) if words else 0.0
        
        # Completeness (sentence structure integrity)
        sentences = [s.strip() for s in doc.content.split('.') if s.strip()]
        complete_sentences = len([s for s in sentences if len(s.split()) >= 3])
        completeness_score = complete_sentences / len(sentences) if sentences else 0.0
        
        # Authority and freshness
        authority_score = doc.metadata.authority_score
        temporal_score = doc.metadata.temporal_relevance
        
        # Compression quality
        compression_quality = 1.0 - doc.metadata.compression_loss
        
        # Weighted combination
        quality_score = (
            0.35 * semantic_score +
            0.20 * density_score +
            0.15 * completeness_score +
            0.15 * authority_score +
            0.10 * temporal_score +
            0.05 * compression_quality
        )
        
        self.quality_cache[cache_key] = quality_score
        return quality_score
    
    def filter_documents_by_quality(self, docs: List[ProcessedDocument], query: str) -> List[ProcessedDocument]:
        """Filter documents by embedding-informed quality assessment"""
        quality_scores = []
        for doc in docs:
            quality = self.assess_document_quality(doc, query)
            quality_scores.append((doc, quality))
        
        filtered_docs = [
            (doc, score) for doc, score in quality_scores 
            if score >= self.quality_threshold
        ]
        
        if not filtered_docs:
            logger.warning(f"No documents met quality threshold {self.quality_threshold}, using top documents")
            filtered_docs = sorted(quality_scores, key=lambda x: x[1], reverse=True)[:3]
        
        filtered_docs.sort(key=lambda x: x[1], reverse=True)
        
        result_docs = []
        for doc, score in filtered_docs:
            doc.processing_info['quality_score'] = score
            doc.processing_info['quality_threshold_met'] = score >= self.quality_threshold
            result_docs.append(doc)
        
        logger.info(f"Quality filtering: {len(docs)} -> {len(result_docs)} documents")
        return result_docs

class DocumentProcessor:
    """Process and optimize documents for prompt construction"""
    
    def __init__(self, config: PromptConstructionConfig):
        self.config = config
        self.embedding_optimizer = EmbeddingInformedOptimizer()
        
    def process_documents(self, raw_docs: List[Dict], query: str, budget: BudgetAllocation) -> List[ProcessedDocument]:
        """Process raw documents into optimized format"""
        processed_docs = []
        for i, raw_doc in enumerate(raw_docs):
            metadata = self._extract_metadata(raw_doc, i)
            content = raw_doc.get('content', str(raw_doc))
            
            proc_doc = ProcessedDocument(
                content=content,
                metadata=metadata,
                processing_info={'original_index': i}
            )
            processed_docs.append(proc_doc)
        
        quality_filtered = self.embedding_optimizer.filter_documents_by_quality(processed_docs, query)
        ordered_docs = self._order_documents(quality_filtered, query)
        budget_constrained = self._apply_budget_constraints(ordered_docs, budget)
        
        return budget_constrained
    
    def _extract_metadata(self, raw_doc: Dict, index: int) -> DocumentMetadata:
        """Extract metadata from raw document"""
        return DocumentMetadata(
            source_id=raw_doc.get('source_id', f'doc_{index}'),
            url=raw_doc.get('url'),
            retrieval_score=raw_doc.get('score', 0.0),
            vector_score=raw_doc.get('vector_score', 0.0),
            compression_ratio=raw_doc.get('compression_ratio', 1.0),
            compression_loss=raw_doc.get('compression_loss', 0.0),
            last_updated=raw_doc.get('last_updated'),
            authority_score=raw_doc.get('authority_score', 0.5),
            temporal_relevance=raw_doc.get('temporal_relevance', 0.5),
            token_count=raw_doc.get('token_count', len(str(raw_doc).split()))
        )
    
    def _order_documents(self, docs: List[ProcessedDocument], query: str) -> List[ProcessedDocument]:
        """Order documents according to configured strategy"""
        if self.config.order_strategy == DocumentOrder.RELEVANCE_SCORE:
            return sorted(docs, key=lambda d: d.metadata.retrieval_score, reverse=True)
        elif self.config.order_strategy == DocumentOrder.COMPRESSION_RETENTION:
            return sorted(docs, key=lambda d: 1.0 - d.metadata.compression_loss, reverse=True)
        elif self.config.order_strategy == DocumentOrder.TEMPORAL:
            return sorted(docs, key=lambda d: d.metadata.temporal_relevance, reverse=True)
        elif self.config.order_strategy == DocumentOrder.AUTHORITY:
            return sorted(docs, key=lambda d: d.metadata.authority_score, reverse=True)
        elif self.config.order_strategy == DocumentOrder.HYBRID_WEIGHTED:
            return self._hybrid_weighted_ordering(docs, query)
        else:
            return docs
    
    def _hybrid_weighted_ordering(self, docs: List[ProcessedDocument], query: str) -> List[ProcessedDocument]:
        """Hybrid weighted document ordering"""
        def compute_hybrid_score(doc: ProcessedDocument) -> float:
            relevance = doc.metadata.retrieval_score
            quality = doc.processing_info.get('quality_score', 0.5)
            authority = doc.metadata.authority_score
            compression_quality = 1.0 - doc.metadata.compression_loss
            temporal = doc.metadata.temporal_relevance
            
            query_lower = query.lower()
            if any(word in query_lower for word in ['recent', 'latest', 'current', 'now']):
                weights = [0.25, 0.20, 0.15, 0.10, 0.30]
            elif any(word in query_lower for word in ['analysis', 'analyze', 'study', 'research']):
                weights = [0.25, 0.30, 0.25, 0.15, 0.05]
            else:
                weights = [0.30, 0.25, 0.20, 0.15, 0.10]
            
            return (
                weights[0] * relevance +
                weights[1] * quality +
                weights[2] * authority +
                weights[3] * compression_quality +
                weights[4] * temporal
            )
        
        scored_docs = [(doc, compute_hybrid_score(doc)) for doc in docs]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        result_docs = []
        for doc, score in scored_docs:
            doc.processing_info['hybrid_score'] = score
            result_docs.append(doc)
        
        return result_docs
    
    def _apply_budget_constraints(self, docs: List[ProcessedDocument], budget: BudgetAllocation) -> List[ProcessedDocument]:
        """Apply token budget constraints to document selection"""
        available_tokens = budget.retrieval_tokens
        selected_docs = []
        used_tokens = 0
        
        for doc in docs:
            doc_tokens = doc.metadata.token_count
            if used_tokens + doc_tokens <= available_tokens:
                selected_docs.append(doc)
                used_tokens += doc_tokens
            else:
                remaining_tokens = available_tokens - used_tokens
                if remaining_tokens > 50:
                    truncated_content = self._truncate_document(doc.content, remaining_tokens)
                    if truncated_content:
                        truncated_doc = ProcessedDocument(
                            content=truncated_content,
                            metadata=doc.metadata,
                            processing_info={
                                **doc.processing_info,
                                'truncated': True,
                                'original_tokens': doc_tokens,
                                'final_tokens': remaining_tokens
                            }
                        )
                        selected_docs.append(truncated_doc)
                        used_tokens += remaining_tokens
                break
        
        logger.info(f"Budget constraint: {len(docs)} -> {len(selected_docs)} docs, {used_tokens}/{available_tokens} tokens used")
        return selected_docs
    
    def _truncate_document(self, content: str, max_tokens: int) -> str:
        """Intelligently truncate document to fit token budget"""
        words = content.split()
        if len(words) <= max_tokens:
            return content
        
        sentences = content.split('.')
        truncated = ""
        word_count = 0
        
        for sentence in sentences:
            sentence_words = sentence.strip().split()
            if word_count + len(sentence_words) <= max_tokens - 1:
                truncated += sentence.strip() + ". "
                word_count += len(sentence_words)
            else:
                break
        
        if not truncated.strip():
            truncated = " ".join(words[:max_tokens])
        
        return truncated.strip()

class PromptTemplateManager:
    """Manage prompt templates with Jinja2 support"""
    
    def __init__(self, templates_dir: Optional[str] = None):
        self.templates_dir = Path(templates_dir) if templates_dir else Path("contextchain/prompts")
        # Verify templates directory exists
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory {self.templates_dir} does not exist, falling back to default templates")
            self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.templates_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        self.custom_filters()
        
        # Verify required template files
        self._verify_templates()

    def custom_filters(self):
        """Add custom Jinja2 filters"""
        def truncate_words(text, max_words=100):
            words = str(text).split()
            if len(words) <= max_words:
                return text
            return " ".join(words[:max_words]) + "..."
        
        def citation_format(source_id):
            return f"[{source_id}]"
        
        def metadata_summary(metadata: DocumentMetadata):
            """Generate summary for DocumentMetadata object"""
            return f"(score: {metadata.retrieval_score:.2f})"
        
        self.env.filters['truncate_words'] = truncate_words
        self.env.filters['cite'] = citation_format
        self.env.filters['meta'] = metadata_summary
    
    def _verify_templates(self):
        """Verify that required .j2 template files exist"""
        template_map = {
            PromptStyle.QA_STRUCTURED: "qa_structured.j2",
            PromptStyle.REASONING_FRIENDLY: "reasoning_friendly.j2",
            PromptStyle.CHAIN_OF_THOUGHT: "chain_of_thought.j2",
            PromptStyle.FEW_SHOT: "few_shot.j2",
            PromptStyle.COMPLIANCE: "compliance.j2",
            PromptStyle.ANALYTICAL: "analytical.j2",
            PromptStyle.COMPARATIVE: "comparative.j2"
        }
        missing_templates = []
        for style, template_file in template_map.items():
            template_path = self.templates_dir / template_file
            if not template_path.exists():
                logger.warning(f"Template file {template_file} not found in {self.templates_dir}")
                missing_templates.append(template_file)
        if missing_templates:
            logger.info(f"Missing templates: {', '.join(missing_templates)}. Will use default templates for these styles.")

    def load_template(self, style: PromptStyle) -> jinja2.Template:
        """Load template for given style"""
        template_map = {
            PromptStyle.QA_STRUCTURED: "qa_structured.j2",
            PromptStyle.REASONING_FRIENDLY: "reasoning_friendly.j2",
            PromptStyle.CHAIN_OF_THOUGHT: "chain_of_thought.j2",
            PromptStyle.FEW_SHOT: "few_shot.j2",
            PromptStyle.COMPLIANCE: "compliance.j2",
            PromptStyle.ANALYTICAL: "analytical.j2",
            PromptStyle.COMPARATIVE: "comparative.j2"
        }
        
        template_file = template_map.get(style, "qa_structured.j2")
        
        try:
            return self.env.get_template(template_file)
        except jinja2.TemplateNotFound:
            logger.warning(f"Template {template_file} not found, using default")
            return self.env.from_string(self._get_default_template(style))
    
    def _get_default_template(self, style: PromptStyle) -> str:
        """Get default template if file not found"""
        templates = {
            PromptStyle.QA_STRUCTURED: """System: You are an expert assistant. Use only the information provided in the context to answer questions accurately and comprehensively.

CONTEXT:
{% for doc in documents %}
--- Document {{ loop.index }} {% if doc.metadata.source_id %}({{ doc.metadata.source_id }}){% endif %} {{ doc.metadata|meta }} ---
{{ doc.content|truncate_words(200) }}
{% if not loop.last %}

{% endif %}
{% endfor %}

QUERY: {{ query }}

INSTRUCTIONS:
- Answer precisely using only the provided context
- Cite sources using square brackets {{ "[source_id]"|cite }}
- If information is insufficient, state what is known and what cannot be determined
- Maintain factual accuracy and avoid speculation""",

            PromptStyle.REASONING_FRIENDLY: """System: You are an analytical expert. Approach this systematically with clear reasoning.

CONTEXT:
{% for doc in documents %}
--- Source {{ loop.index }}: {{ doc.metadata.source_id }} {{ doc.metadata|meta }} ---
{{ doc.content }}
{% if not loop.last %}

{% endif %}
{% endfor %}

QUERY: {{ query }}

APPROACH:
1. First, identify the key information relevant to the query
2. Analyze the relationships and patterns in the data
3. Synthesize insights based on the evidence
4. Provide a clear, reasoned conclusion

Please work through this step-by-step, citing your sources {{ "[source_id]"|cite }}.""",

            PromptStyle.CHAIN_OF_THOUGHT: """System: Think through this step-by-step, showing your reasoning process clearly.

CONTEXT:
{% for doc in documents %}
--- Document {{ loop.index }}: {{ doc.metadata.source_id }} ---
{{ doc.content }}

{% endfor %}

QUERY: {{ query }}

Let me think about this step by step:

Step 1: What does the query ask for?
Step 2: What relevant information do I have?
Step 3: How do the pieces connect?
Step 4: What can I conclude?

Please follow this structured approach and cite sources {{ "[source_id]"|cite }} at each step.""",

            PromptStyle.ANALYTICAL: """System: You are conducting a thorough analysis. Be systematic, evidence-based, and comprehensive.

CONTEXT ANALYSIS:
{% for doc in documents %}
Source {{ loop.index }}: {{ doc.metadata.source_id }} {{ doc.metadata|meta }}
Content: {{ doc.content }}
Key Points: [To be identified]

{% endfor %}

ANALYTICAL QUERY: {{ query }}

ANALYTICAL FRAMEWORK:
1. Data Review: Summarize key information from sources
2. Pattern Identification: Look for trends, relationships, anomalies
3. Evidence Synthesis: Combine insights from multiple sources
4. Conclusions: Draw evidence-based conclusions
5. Limitations: Note any gaps or uncertainties

Proceed with systematic analysis, citing sources {{ "[source_id]"|cite }} throughout."""
        }
        
        return templates.get(style, templates[PromptStyle.QA_STRUCTURED])

class SafetyFilter:
    """Content safety and compliance filtering"""
    
    def __init__(self, config: PromptConstructionConfig):
        self.config = config
        self.pii_patterns = self._initialize_pii_patterns()
        self.safety_keywords = self._initialize_safety_keywords()
    
    def _initialize_pii_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize PII detection patterns"""
        return {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b')
        }
    
    def _initialize_safety_keywords(self) -> List[str]:
        """Initialize safety filtering keywords"""
        return [
            'password', 'secret', 'confidential', 'classified',
            'internal only', 'proprietary', 'restricted'
        ]
    
    def filter_content(self, content: str) -> Tuple[str, List[str]]:
        """Filter content for PII and safety issues"""
        filtered_content = content
        issues_found = []
        
        if self.config.safety_filtering:
            for pii_type, pattern in self.pii_patterns.items():
                matches = pattern.findall(filtered_content)
                if matches:
                    issues_found.append(f"pii_{pii_type}")
                    filtered_content = pattern.sub(f'[{pii_type.upper()}_REDACTED]', filtered_content)
            
            content_lower = filtered_content.lower()
            for keyword in self.safety_keywords:
                if keyword in content_lower:
                    issues_found.append(f"safety_keyword_{keyword.replace(' ', '_')}")
                    if self.config.compliance_mode:
                        sentences = filtered_content.split('.')
                        filtered_sentences = []
                        for sentence in sentences:
                            if keyword not in sentence.lower():
                                filtered_sentences.append(sentence)
                            else:
                                filtered_sentences.append('[CONTENT_FILTERED_FOR_COMPLIANCE]')
                        filtered_content = '.'.join(filtered_sentences)
        
        return filtered_content, issues_found

class ContextEngineer:
    """
    Main Context Engineer class
    Integrates all components for advanced prompt construction
    """
    
    def __init__(self, config: Optional[PromptConstructionConfig] = None, templates_dir: Optional[str] = None):
        self.config = config or PromptConstructionConfig()
        self.templates_dir = templates_dir or "/Users/mohammednihal/Desktop/ContextChain/ContextChain/contextchain/prompts"
        self.doc_processor = DocumentProcessor(self.config)
        self.template_manager = PromptTemplateManager(self.templates_dir)
        self.safety_filter = SafetyFilter(self.config)
        
        # Token estimation (simplified)
        self.avg_chars_per_token = 4
        
        logger.info(f"ContextEngineer initialized with style: {self.config.style.value}, templates_dir: {self.templates_dir}")
    
    async def build_prompt(self, 
                          query: str,
                          raw_docs: List[Dict],
                          budget: BudgetAllocation,
                          semantic_state: Dict[str, Any] = None,
                          complexity: Optional[QueryComplexity] = None) -> str:
        """
        Main async method: Build optimized prompt from query and documents
        """
        start_time = datetime.utcnow()
        try:
            # Process documents asynchronously
            processed_docs = await asyncio.to_thread(
                self.doc_processor.process_documents,
                raw_docs, query, budget
            )
            
            # Apply safety filtering
            filtered_docs = []
            total_safety_issues = []
            for doc in processed_docs:
                filtered_content, issues = await asyncio.to_thread(self.safety_filter.filter_content, doc.content)
                total_safety_issues.extend(issues)
                filtered_doc = ProcessedDocument(
                    content=filtered_content,
                    metadata=doc.metadata,
                    processing_info={
                        **doc.processing_info,
                        'safety_issues': issues,
                        'safety_filtered': len(issues) > 0
                    }
                )
                filtered_docs.append(filtered_doc)
            
            # Load and render template
            template = await asyncio.to_thread(self.template_manager.load_template, self.config.style)
            template_context = self._prepare_template_context(query, filtered_docs, budget, complexity)
            rendered_prompt = await asyncio.to_thread(template.render, **template_context)
            
            # Token budget validation
            estimated_tokens = self._estimate_tokens(rendered_prompt)
            if estimated_tokens > self.config.max_prompt_tokens:
                logger.warning(f"Prompt exceeds token limit: {estimated_tokens} > {self.config.max_prompt_tokens}")
                rendered_prompt = self._truncate_prompt(rendered_prompt, self.config.max_prompt_tokens)
            
            # Add metadata footer
            if self.config.include_metadata:
                metadata_footer = self._generate_metadata_footer(
                    processed_docs, total_safety_issues, estimated_tokens, start_time
                )
                rendered_prompt += "\n\n" + metadata_footer
            
            logger.info(f"Prompt built successfully: {estimated_tokens} tokens, "
                       f"{len(filtered_docs)} docs, {len(total_safety_issues)} safety issues")
            
            return rendered_prompt.strip()
            
        except Exception as e:
            logger.error(f"Error building prompt: {str(e)}")
            return self._build_fallback_prompt(query, raw_docs)
    
    async def close(self):
        """Close resources and clear caches asynchronously"""
        try:
            self.doc_processor.embedding_optimizer.quality_cache.clear()
            logger.info("ContextEngineer caches cleared successfully")
        except Exception as e:
            logger.error(f"Error closing ContextEngineer: {str(e)}")
    
    def _prepare_template_context(self, query: str, docs: List[ProcessedDocument], budget: BudgetAllocation, complexity: Optional[QueryComplexity]) -> Dict[str, Any]:
        """Prepare context variables for template rendering"""
        context = {
            'query': query,
            'documents': docs,
            'doc_count': len(docs),
            'budget': asdict(budget),  # Convert BudgetAllocation dataclass to dict
            'style': self.config.style.value,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        if complexity:
            context['complexity'] = asdict(complexity)  # Convert QueryComplexity dataclass to dict
        
        if self.config.style == PromptStyle.FEW_SHOT:
            context['examples'] = self._get_few_shot_examples(query, self.config.few_shot_examples)
        
        if self.config.compliance_mode:
            context['compliance'] = {
                'enabled': True,
                'safety_level': 'high',
                'pii_redaction': True
            }
        
        return context
    
    def _get_few_shot_examples(self, query: str, num_examples: int) -> List[Dict[str, str]]:
        """Get few-shot examples relevant to query (simplified implementation)"""
        query_lower = query.lower()
        
        if 'analyze' in query_lower or 'analysis' in query_lower:
            examples = [
                {
                    'query': 'Analyze the sales performance for Q1 2025',
                    'context': 'Q1 2025 sales: $2.3M (+15% YoY). Key drivers: new product launch, market expansion.',
                    'response': 'Q1 2025 sales reached $2.3M, representing a strong 15% year-over-year growth [source_1]. The primary growth drivers were the successful new product launch and strategic market expansion initiatives [source_1].'
                },
                {
                    'query': 'Analyze customer satisfaction trends',
                    'context': 'Customer satisfaction: Q1: 85%, Q2: 82%, Q3: 88%. Main issues: delivery delays, support response times.',
                    'response': 'Customer satisfaction shows a recovery pattern: declining from 85% in Q1 to 82% in Q2, then rebounding to 88% in Q3 [source_1]. Key improvement areas identified were delivery delays and support response times [source_1].'
                }
            ]
        elif 'compare' in query_lower or 'versus' in query_lower:
            examples = [
                {
                    'query': 'Compare product performance across regions',
                    'context': 'Product A: North America $1.2M, Europe $800K, Asia $600K. Product B: North America $900K, Europe $1.1M, Asia $400K.',
                    'response': 'Product A shows stronger performance in North America ($1.2M) compared to Product B ($900K), while Product B leads in Europe ($1.1M vs $800K) [source_1]. Both products show similar patterns in Asia with Product A slightly ahead [source_1].'
                }
            ]
        else:
            examples = [
                {
                    'query': 'What are the key findings from the recent report?',
                    'context': 'Report findings: 1) Revenue increased 12%, 2) Customer base grew by 8%, 3) Market share expanded to 15%.',
                    'response': 'The recent report highlights three key achievements: revenue growth of 12%, customer base expansion of 8%, and market share increase to 15% [source_1].'
                }
            ]
        
        return examples[:num_examples]
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (simplified implementation)"""
        return len(text) // self.avg_chars_per_token
    
    def _truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        """Truncate prompt to fit token budget"""
        max_chars = max_tokens * self.avg_chars_per_token
        if len(prompt) <= max_chars:
            return prompt
        
        paragraphs = prompt.split('\n\n')
        truncated = ""
        
        for paragraph in paragraphs:
            if len(truncated + paragraph) <= max_chars:
                truncated += paragraph + "\n\n"
            else:
                break
        
        if not truncated.strip():
            truncated = prompt[:max_chars]
        
        return truncated.strip() + "\n\n[PROMPT_TRUNCATED_TO_FIT_BUDGET]"
    
    def _generate_metadata_footer(self, docs: List[ProcessedDocument], safety_issues: List[str], estimated_tokens: int, start_time: datetime) -> str:
        """Generate metadata footer for transparency"""
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        footer_lines = [
            "---",
            "PROCESSING METADATA:",
            f"- Documents processed: {len(docs)}",
            f"- Estimated tokens: {estimated_tokens}",
            f"- Processing time: {processing_time:.3f}s",
            f"- Safety issues detected: {len(safety_issues)}",
            f"- Style: {self.config.style.value}",
            f"- Quality threshold applied: {self.doc_processor.embedding_optimizer.quality_threshold}"
        ]
        
        if safety_issues:
            footer_lines.append(f"- Safety filters applied: {', '.join(set(safety_issues))}")
        
        if self.config.include_provenance:
            footer_lines.append("- Source provenance available for all citations")
        
        return "\n".join(footer_lines)
    
    def _build_fallback_prompt(self, query: str, raw_docs: List[Dict]) -> str:
        """Build simple fallback prompt when main process fails"""
        context_parts = []
        for i, doc in enumerate(raw_docs[:3]):
            content = doc.get('content', str(doc))[:500]
            context_parts.append(f"Document {i+1}: {content}")
        
        context = "\n\n".join(context_parts)
        return f"""Context:
{context}

Query: {query}

Please answer based on the provided context."""

def test_context_engineer():
    """Test the ContextEngineer functionality"""
    config = PromptConstructionConfig(
        style=PromptStyle.ANALYTICAL,
        max_prompt_tokens=1500,
        include_metadata=True,
        include_provenance=True,
        safety_filtering=True
    )
    
    engineer = ContextEngineer(config, templates_dir="/Users/mohammednihal/Desktop/ContextChain/ContextChain/contextchain/prompts")
    
    query = "Analyze the Q3 2025 sales performance and identify key growth drivers"
    raw_docs = [
        {
            'content': 'Q3 2025 sales reached $2.8M, representing a 22% increase from Q2 2025 ($2.3M). The primary growth drivers were the launch of our premium product line, expansion into the European market, and improved customer retention rates.',
            'source_id': 'sales_report_q3_2025',
            'score': 0.95,
            'authority_score': 0.9,
            'temporal_relevance': 0.95
        },
        {
            'content': 'European market expansion contributed $400K to Q3 revenue. Customer acquisition cost decreased by 15% due to improved marketing efficiency. Premium product line generated 30% of total revenue.',
            'source_id': 'market_analysis_europe',
            'score': 0.88,
            'authority_score': 0.85,
            'temporal_relevance': 0.9
        },
        {
            'content': 'Customer retention improved from 78% in Q2 to 85% in Q3. This improvement was attributed to enhanced customer support and product quality improvements.',
            'source_id': 'customer_metrics_q3',
            'score': 0.82,
            'authority_score': 0.8,
            'temporal_relevance': 0.92
        }
    ]
    
    from .acba import BudgetAllocation, BudgetArm
    
    budget = BudgetAllocation(
        retrieval_tokens=800,
        compression_tokens=200,
        generation_tokens=500,
        total_budget=1500,
        arm_selected=BudgetArm.ADAPTIVE_COMPRESS,
        confidence_score=0.85,
        hierarchy_weights={},
        expected_utility=0.78,
        allocation_timestamp=datetime.utcnow()
    )
    
    # Run async test
    prompt = asyncio.run(engineer.build_prompt(query, raw_docs, budget))
    
    print("Generated Prompt:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    
    asyncio.run(engineer.close())
    return prompt

if __name__ == "__main__":
    test_context_engineer()