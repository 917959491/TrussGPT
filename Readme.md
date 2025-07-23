# TrussGPT
TrussGPT is a large language model (LLM)-driven generative framework for the inverse design of truss metamaterials. It integrates discrete structural tokenization, property-conditioned generation, and natural language interaction to enable efficient, accurate, and interpretable design of mechanical metamaterials.

# 🚀 Key Features
```
1、Inverse design from mechanical properties (e.g., Young’s modulus, Poisson’s ratio)
2、Structure generation based on full stress–strain curves
3、Natural language interaction for intuitive design and analysis
4、Significantly faster and more accurate than traditional optimization-based methods
```

# 🛠️ Key Dependencies
Ensure the following core packages are installed to avoid compatibility issues:

```
Package	Version
torch	2.2.0+cu118
transformers	4.51.3
accelerate	1.6.0
streamlit	1.44.1
llamafactory	0.9.3.dev0
einops	0.8.1
scikit-learn	1.6.1
numpy	1.24.4
pandas	2.2.3
matplotlib	3.10.1
sentencepiece	0.2.0
```

# 📁 Model Weight Structure (model/)
The model weights required for this project are organized under the model/ directory, which includes two submodules for Truss-Tokenizer and TrussGPT-Designer.

```
model/
├── Truss-Tokenizer/
│   ├── best_model.pt         
│   └── best_c_model.pt        
├── TrussGPT-Designer/
│   ├── E11E22E33.pt          
│   ├── v21.pt                 
│   ├── E33.pt                
│   ├── SS.pt                
│   ├── E33v21.pt              
│   └── E11E22E33v21.pt        
```

# 🔗 Download Trained Weights 
You can download all model weights from the following link:

📦 Baidu Netdisk:
```
https://pan.baidu.com/s/1UEIwbZD71x-gqlSIMIcScg 
Extraction Code: zjpi 
``` 

After downloading, please place all .pt files into their respective directories under model/Truss-Tokenizer/ and model/TrussGPT-Designer/.

# 🔧 Quick Start
Launch the Streamlit-based web demo:

```
cd llm_script/web_demo
streamlit run main.py
```

