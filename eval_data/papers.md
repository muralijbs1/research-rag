# Test Papers

Papers used for development and evaluation of this project.
Download each PDF and save it to `uploaded_pdfs/` before ingesting.

| Paper | Authors | Year | arXiv Link |
|---|---|---|---|
| Attention Is All You Need | Vaswani et al. | 2017 | [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) |
| Agentic RAG: A Survey | Singh et al. | 2025 | [arxiv.org/abs/2501.09136](https://arxiv.org/abs/2501.09136) |
| AutoGen: Enabling Next-Gen LLM Applications | Wu et al. | 2023 | [arxiv.org/abs/2308.08155](https://arxiv.org/abs/2308.08155) |
| Language Models are Few-Shot Learners (GPT-3) | Brown et al. | 2020 | [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165) |
| Mixture-of-Agents Enhances LLM Capabilities | Wang et al. | 2024 | [arxiv.org/abs/2406.04692](https://arxiv.org/abs/2406.04692) |
| ReAct: Synergizing Reasoning and Acting | Yao et al. | 2022 | [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629) |
| RAG for Knowledge-Intensive NLP Tasks | Lewis et al. | 2020 | [arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401) |
| Self-Consistency Improves Chain of Thought | Wang et al. | 2022 | [arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171) |
| The AI Scientist | Lu et al. | 2024 | [arxiv.org/abs/2408.06292](https://arxiv.org/abs/2408.06292) |
| TinyLlama: An Open-Source Small Language Model | Zhang et al. | 2024 | [arxiv.org/abs/2401.02385](https://arxiv.org/abs/2401.02385) |
| Toolformer: Language Models Can Teach Themselves to Use Tools | Schick et al. | 2023 | [arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761) |

## How to Download

1. Click any link above
2. Click the **PDF** button on the arXiv page
3. Save to `uploaded_pdfs/` in your project root
4. Run:
```python
   from src.ingestion.ingest_pipeline import run_ingestion
   run_ingestion("uploaded_pdfs/your_file.pdf")
```