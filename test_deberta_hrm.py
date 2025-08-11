#!/usr/bin/env python3

import torch
import sys
import os

# Add HRM to path
sys.path.append('/Users/erik/Dev/rkstgr/hrm-experiments/HRM')

from models.hrm.deberta_hrm import DebertaHRM, DebertaHRMConfig
from ai2arc_dataset import AI2ArcDatasetConfig, create_ai2arc_dataloader

def test_deberta_hrm_model():
    """Test DebertaHRM model loading and forward pass"""
    print("🧪 Testing DebertaHRM model...")
    
    # Create model config
    model_config = DebertaHRMConfig(
        model_name="microsoft/deberta-v3-base",
        hidden_size=768,
        num_labels=4,
        H_cycles=1,
        L_cycles=2,
        H_layers=1, 
        L_layers=1
    )
    
    print(f"📋 Model config: {model_config}")
    
    # Create model
    print("🏗️  Creating DebertaHRM model...")
    model = DebertaHRM(model_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Parameters: {trainable_params:,} trainable / {total_params:,} total")
    print(f"📊 Trainable ratio: {trainable_params/total_params:.3f}")
    
    # Test with sample data
    print("🔬 Testing forward pass...")
    
    # Create sample batch (similar to what AI2ArcDataset produces)
    batch_size = 2
    seq_len = 46  # From our dataset test
    num_choices = 4
    
    sample_batch = {
        'input_ids': torch.randint(0, 1000, (batch_size, num_choices, seq_len)),
        'attention_mask': torch.ones(batch_size, num_choices, seq_len, dtype=torch.long),
        'token_type_ids': torch.zeros(batch_size, num_choices, seq_len, dtype=torch.long),
        'labels': torch.randint(0, 4, (batch_size,))
    }
    
    print(f"🔍 Sample batch shapes:")
    for k, v in sample_batch.items():
        print(f"  {k}: {v.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**sample_batch)
    
    print(f"✅ Forward pass successful!")
    
    if isinstance(outputs, tuple) and len(outputs) == 2:
        loss, logits = outputs
        print(f"📈 Loss: {loss.item():.4f}")
        print(f"📈 Logits shape: {logits.shape}")
        print(f"📈 Logits sample: {logits[0]}")
        
        # Test predictions
        preds = torch.argmax(logits, dim=-1)
        print(f"🎯 Predictions: {preds}")
        print(f"🎯 Labels: {sample_batch['labels']}")
        
    else:
        print(f"❓ Unexpected output format: {type(outputs)}")
    
    return True

def test_with_real_data():
    """Test with real AI2ARC data"""
    print("\n🧪 Testing with real AI2ARC data...")
    
    # Create dataset
    dataset_config = AI2ArcDatasetConfig(batch_size=2, num_workers=0)
    dataloader = create_ai2arc_dataloader(dataset_config, "train")
    
    # Get one real batch
    real_batch = next(iter(dataloader))
    print(f"🔍 Real batch shapes:")
    for k, v in real_batch.items():
        print(f"  {k}: {v.shape}")
    
    # Create model
    model_config = DebertaHRMConfig()
    model = DebertaHRM(model_config)
    
    # Forward pass with real data
    print("🔬 Testing forward pass with real data...")
    with torch.no_grad():
        outputs = model(**real_batch)
    
    if isinstance(outputs, tuple) and len(outputs) == 2:
        loss, logits = outputs
        print(f"✅ Real data forward pass successful!")
        print(f"📈 Loss: {loss.item():.4f}")
        print(f"📈 Logits shape: {logits.shape}")
        
        # Test predictions
        preds = torch.argmax(logits, dim=-1)
        print(f"🎯 Predictions: {preds}")
        print(f"🎯 Labels: {real_batch['labels']}")
        
        # Calculate accuracy
        accuracy = (preds == real_batch['labels']).float().mean()
        print(f"🎯 Accuracy: {accuracy:.3f}")
        
    return True

if __name__ == "__main__":
    print("🚀 Starting DebertaHRM model tests...\n")
    
    try:
        # Test model creation and forward pass
        test_deberta_hrm_model()
        
        # Test with real data
        test_with_real_data()
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)