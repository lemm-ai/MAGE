# MAGE Enhancement Design Document

## Overview
This document outlines the comprehensive redesign of MAGE to integrate advanced AI music generation capabilities, cross-platform GPU support, and a professional Gradio-based GUI.

## Critical Dependencies Analysis

### 1. LyricMind-AI
- **Source**: https://github.com/AmirHaytham/LyricMind-AI
- **Requirements**: Python 3.8+, PyTorch, LSTM-based model
- **License**: MIT
- **Compatibility**: ✅ Compatible with Python 3.10
- **Model**: Pretrained LSTM model (best_model.pth, vocab.json required)
- **Integration Point**: Replace procedural generation in lyrics module

### 2. Singing Quality Enhancement
- **Source**: https://github.com/wimmerb/singing-quality-enhancement
- **Requirements**: Python 3.x, PyTorch 1.10.x, torchaudio 0.10.x
- **License**: Apache 2.0
- **Architecture**: FullSubNet (denoising)
- **Compatibility**: ⚠️ Need to test with PyTorch 2.0
- **Models**: Pretrained in Experiments/Denoising folder
- **Integration Point**: Vocal enhancement post-processing

### 3. MusicControlNet Research Status
- **Status**: ⚠️ No public MusicControlNet/MusicControlNet repository found
- **Alternative Approaches**:
  - Feature-based style matching
  - Prompt conditioning with extracted parameters
  - Spectral characteristic alignment
- **Implementation**: Custom style consistency module

### 4. Demucs
- **Source**: https://github.com/facebookresearch/demucs
- **Requirements**: Python 3.8+, PyTorch, torchaudio
- **License**: MIT
- **Compatibility**: ✅ Fully compatible
- **Integration Point**: Stem separation

### 5. Pedalboard
- **Source**: Spotify's Pedalboard
- **License**: GPL-3.0 (⚠️ Compatible for library usage)
- **Compatibility**: ✅ Cross-platform including Linux
- **Integration Point**: Professional EQ effects

## GPU Support Strategy

### Enhanced Multi-Platform Detection
```python
def get_optimal_device():
    if torch.cuda.is_available():
        # Works for both NVIDIA (CUDA) and AMD (ROCm)
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"
```

**Note**: PyTorch with ROCm uses same CUDA API but requires ROCm-specific build.

## Implementation Phases

### Phase 1: Foundation & GPU (Current)
- Add cross-platform GPU detection
- Test on WSL/Ubuntu
- Update configuration system

### Phase 2: Lyrics & Stems
- Integrate LyricMind-AI
- Add Demucs stem separation
- Create timeline system

### Phase 3: Enhancement
- Add singing-quality-enhancement
- Implement Pedalboard EQ
- Create vocal cleanup workflow

### Phase 4: GUI
- Build Gradio interface
- Add waveform visualization
- Create EQ/enhancement popup

### Phase 5: Style Consistency
- Implement feature extraction
- Add style matching
- Optimize clip chaining

## Risk Assessment

### Critical Risks
1. **MusicControlNet unavailable** → Use feature-based approach
2. **PyTorch version conflicts** → Test compatibility
3. **GPL licensing** → Pedalboard is library usage (OK)
4. **Model sizes** → Implement lazy loading
5. **WSL audio** → Use cross-platform libs (soundfile)

## Next Actions
1. Update GPU detection
2. Add new module structure
3. Integrate dependencies one by one
4. Test each phase before proceeding
