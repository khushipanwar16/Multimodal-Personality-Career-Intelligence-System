## Multimodal Personality-Driven Career Intelligence System
## Overview

An AI-powered system that predicts user personality using multimodal inputs (Text + Image) and recommends suitable careers through semantic similarity and embedding-based matching.

The system combines NLP, computer vision, and embedding-based retrieval to create an intelligent and explainable career recommendation experience.

## Tech Stack

Python

Hugging Face Transformers (DistilBERT)

SentenceTransformers (MiniLM embeddings)

PyTorch

Scikit-learn

DeepFace (Image Emotion Analysis)

Plotly (Interactive Visualization)

Streamlit (Frontend UI)

## Dataset

MBTI Personality Dataset (Kaggle) — Text-based personality training

Custom Career Dataset — Career descriptions for semantic matching

## Current Progress
## NLP & Personality Modeling

Dataset preprocessing & cleaning

Label encoding (MBTI classes)

DistilBERT fine-tuning for personality prediction

Softmax probability generation for confidence scoring

## Career Intelligence Engine

Career embedding generation using SentenceTransformers

Cosine similarity–based career matching

Top-K career recommendation system

3D PCA visualization of career embedding space

## Multimodal Expansion

Image upload support

Emotion-based personality signal extraction (DeepFace)

Text + Image fusion scoring (weighted multimodal fusion)

Final personality prediction from fusion probabilities

## AI Reliability & Validation

Input validation for meaningful personality text

Personality signal strength checking

Confidence-based filtering (LOW / MEDIUM / HIGH confidence levels)

Low-confidence rejection for unreliable inputs

## UI / UX (Streamlit)

Modern animated gradient background

Glassmorphism result cards

Hover effects and fade-in animations

Animated confidence progress bars

Expandable AI reasoning sections

Interactive 3D visualization dashboard

## Project Goal

To build a realistic multimodal AI system that combines NLP, Computer Vision, and semantic intelligence for explainable and human-friendly career recommendations.