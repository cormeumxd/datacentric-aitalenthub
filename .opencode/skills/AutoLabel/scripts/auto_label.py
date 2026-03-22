#!/usr/bin/env python3
"""
Auto-labeling script for news topic classification
Uses zero-shot classification with transformers
"""

import argparse
import pandas as pd
import torch
from transformers import pipeline
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Label mapping for the dataset
LABEL_MAP = {
    0: 'World',
    1: 'Sports',
    2: 'Business',
    3: 'Sci/Tech'
}

# Candidate labels for zero-shot classification
CANDIDATE_LABELS = ['World news', 'Sports', 'Business', 'Science and Technology']

def parse_args():
    parser = argparse.ArgumentParser(description='Auto-label news topics')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--model', default='facebook/bart-large-mnli', help='Zero-shot model')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to process')
    parser.add_argument('--confidence-threshold', type=float, default=0.5, help='Confidence threshold')
    return parser.parse_args()

def main():
    args = parse_args()
    
    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    
    if args.max_samples:
        df = df.head(args.max_samples)
    
    logger.info(f"Loaded {len(df)} samples")
    
    # Check if labels already exist
    if 'theme' in df.columns and df['theme'].notna().any():
        logger.info("Labels already exist. Adding confidence scores.")
        df['confidence'] = 1.0
        df['auto_label'] = df['theme']
        df['model_version'] = 'pre-labeled'
        df.to_csv(args.output, index=False)
        logger.info(f"Saved to {args.output}")
        return
    
    # Initialize zero-shot classifier
    logger.info(f"Loading model: {args.model}")
    classifier = pipeline(
        "zero-shot-classification",
        model=args.model,
        device=0 if torch.cuda.is_available() else -1
    )
    
    logger.info("Starting auto-labeling...")
    
    labels = []
    confidences = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Labeling"):
        text = str(row['text'])[:512]  # Truncate for faster processing
        
        try:
            result = classifier(text, CANDIDATE_LABELS, multi_label=False)
            labels.append(result['labels'][0])
            confidences.append(result['scores'][0])
        except Exception as e:
            logger.warning(f"Error processing row {idx}: {e}")
            labels.append('Unknown')
            confidences.append(0.0)
    
    df['auto_label'] = labels
    df['confidence'] = confidences
    df['model_version'] = args.model
    
    # Save
    df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(df)} labeled samples to {args.output}")
    
    # Statistics
    logger.info("\nLabel distribution:")
    for label, count in df['auto_label'].value_counts().items():
        logger.info(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    
    logger.info(f"\nAverage confidence: {df['confidence'].mean():.3f}")
    logger.info(f"Low confidence samples (< {args.confidence_threshold}): {(df['confidence'] < args.confidence_threshold).sum()}")

if __name__ == '__main__':
    main()
