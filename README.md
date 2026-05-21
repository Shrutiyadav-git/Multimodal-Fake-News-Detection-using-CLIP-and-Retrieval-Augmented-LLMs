# Multimodal Fake News Detection using CLIP and Retrieval-Augmented LLMs

A multimodal fake news detection framework that combines textual, visual, and external knowledge sources to improve misinformation detection. The system integrates BLIP-based image captioning, CLIP similarity analysis, Retrieval-Augmented Generation (RAG), and Gemini LLM reasoning to generate contextual evidence and semantic understanding for news verification.

The final binary and multi-class predictions are produced using a multimodal fusion network that combines:
- Image feature vectors
- Text feature vectors
- CLIP similarity scores
- LLM confidence scores
- LLM predictions
- LLM-generated justifications

This fusion-based architecture enables the model to leverage both semantic reasoning and multimodal feature learning for more reliable fake news classification.

## Key Features
- Multimodal fake news detection using text and images
- BLIP-based image caption generation
- Evidence retrieval using NewsAPI and Wikipedia APIs
- Semantic filtering using Sentence Transformers and cosine similarity
- Gemini LLM-based contextual reasoning and justification generation
- CLIP-based image-text similarity analysis
- Multimodal fusion network for 2-way and 3-way classification
- Confidence score generation and explainable predictions
- Robust API handling with retry and rate-limit management

## Model Architecture
- Text Encoder: BERT
- Image Encoder: Vision Transformer (ViT)
- Image Captioning: BLIP
- Multimodal Alignment: CLIP
- External Knowledge Retrieval: RAG Pipeline
- LLM Reasoning: Gemini API
- Fusion Network: Combines multimodal embeddings, CLIP scores, and LLM outputs for final classification

## Tech Stack
Python, PyTorch, Transformers, BLIP, CLIP, Gemini API, RAG, Sentence Transformers, BERT, ViT, OpenCV, scikit-learn
