import torch
import os

# PATH DEI MODELLI FORD
encoder_path = "checkpoints_ford/encoder/encoder_2003.pth"
classifier_path = "checkpoints_ford/classifier/classifier_2003.pth"

print("="*50)
print("ANALISI MODELLI FORD")
print("="*50)

# ENCODER
if os.path.exists(encoder_path):
    encoder_state = torch.load(encoder_path, map_location='cpu')
    encoder_params = sum(p.numel() for p in encoder_state.values())
    encoder_size_kb = encoder_params * 4 / 1024
    
    print(f"\nENCODER:")
    print(f"  Parametri: {encoder_params:,}")
    print(f"  Peso: {encoder_size_kb:.2f} KB ({encoder_size_kb/1024:.3f} MB)")
    print(f"  File: {os.path.getsize(encoder_path)/1024:.2f} KB")
else:
    print(f"\nENCODER: file non trovato")

# CLASSIFIER
if os.path.exists(classifier_path):
    classifier_state = torch.load(classifier_path, map_location='cpu')
    classifier_params = sum(p.numel() for p in classifier_state.values())
    classifier_size_kb = classifier_params * 4 / 1024
    
    print(f"\nCLASSIFIER:")
    print(f"  Parametri: {classifier_params:,}")
    print(f"  Peso: {classifier_size_kb:.2f} KB ({classifier_size_kb/1024:.3f} MB)")
    print(f"  File: {os.path.getsize(classifier_path)/1024:.2f} KB")
else:
    print(f"\nCLASSIFIER: file non trovato")

# TOTALE
if os.path.exists(encoder_path) and os.path.exists(classifier_path):
    total_params = encoder_params + classifier_params
    total_kb = total_params * 4 / 1024
    
    print(f"\nTOTALE:")
    print(f"  Parametri: {total_params:,}")
    print(f"  Peso: {total_kb:.2f} KB ({total_kb/1024:.3f} MB)")