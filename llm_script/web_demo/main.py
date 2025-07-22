from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st
import numpy as np
from model import vaeModel, GPT, cModel
import json
import re
import pandas as pd
from io import BytesIO

import warnings
warnings.filterwarnings("ignore")

def smart_fix_json(raw: str):
    """
    Convert "approximate JSON" into valid JSON.
    Can fix: trailing backticks `, single quotes ', missing quotes or braces, and concatenated objects like }{.
    Returns a dict on success, or None on failure.
    """
    txt = raw.strip()

    # 0) Remove markdown ```json ``` fences and trailing backticks
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?", "", txt).strip()
    txt = txt.rstrip("`").strip()          # ← Remove trailing standalone backticks

    # 1) If `}{` appears consecutively, keep only the first segment
    if "}{​" in txt:
        txt = txt.split("}{")[0] + "}"

    # 2) Replace trailing single quotes with double quotes
    if txt.endswith("'"):
        txt = txt[:-1] + '"'

    # 3) Append a closing brace '}' if it is missing at the end
    if not txt.endswith("}"):
        txt += "}"

    # 4) If the total number of quotation marks is odd, one is missing → add it before the last closing brace `}`
    if txt.count('"') % 2 == 1:
        last_brace = txt.rfind("}")
        txt = txt[:last_brace] + '"' + txt[last_brace:]

    # 5) Fix inner blocks like "return": "..." which may be missing a closing brace
    m = re.search(r'"return"\s*:\s*"((?:\\.|[^"\\])*)"', txt)
    if m:
        inner = m.group(1)
        if inner.count('{') > inner.count('}'):
            inner_fixed = inner + '}'
            txt = txt.replace(inner, inner_fixed, 1)

    # 6) Ensure balanced outer braces (ignoring those inside strings)
    open_b = close_b = 0
    in_str = esc = False
    for ch in txt:
        if esc: esc = False; continue
        if ch == '\\': esc = True; continue
        if ch == '"': in_str = not in_str; continue
        if not in_str:
            if ch == '{': open_b += 1
            elif ch == '}': close_b += 1
    while close_b > open_b:           # 多余 }
        idx = txt.rfind('}')
        txt = txt[:idx] + txt[idx+1:]
        close_b -= 1
    if open_b > close_b:              # 缺 }
        txt += '}' * (open_b - close_b)

    # 7) Attempt to parse as JSON
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return None

def parse_analyse(analyse: str):
    """
    Parse the output of a large language model and return a (category, result, json_data) tuple.
    Always ensure that category and result have at least default values.
    """
    
    def trim_at_category_other(raw_output):
        """
        If the string contains '{"category": "other"', keep only the part before it.
        Otherwise, return the original content.
        """
        flag = '{"category": "other"'
        index = raw_output.find(flag)
        if index != -1:
            return raw_output[:index].strip()
        return raw_output.strip()

    # Default values (used when parsing fails)
    category = 'other'
    result   = trim_at_category_other(analyse)
    json_data = None

    try:
        # 1) First extract from ```json``` code block
        json_data = clean_and_load_markdown_json(analyse)

        # 2) Directly use json.loads
        if json_data is None:
            json_data = json.loads(analyse)

    except json.JSONDecodeError:
        # 3) Call smart JSON fixer
        json_data = smart_fix_json(analyse)

    except Exception as e:
        print("A non-JSON error occurred during parsing：", e)

    # 4) If successfully parsed as a dict, extract fields
    if isinstance(json_data, dict):
        category = json_data.get('category', 'other')
        result   = json_data.get('return', '')

    return category, result, json_data

torch.cuda.empty_cache()
Truss_Tokenizer_savedFolder = '/home/ljx/Ljx/Materials predict/Truss/TrussGPT/model/Truss-Tokenizer'
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

def plot_truss_structure_transparent(nodes, edges):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Draw nodes using cross markers
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2],
               s=20, c='black', marker='x', linewidths=1.5)

    # Draw edges
    for edge in edges:
        x = [nodes[edge[0], 0], nodes[edge[1], 0]]
        y = [nodes[edge[0], 1], nodes[edge[1], 1]]
        z = [nodes[edge[0], 2], nodes[edge[1], 2]]
        ax.plot(x, y, z, color='steelblue', linewidth=2)

    ax.axis('off')
    ax.set_box_aspect([1, 1, 1])

    # Save to memory
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True,dpi=350)
    plt.close(fig)
    buf.seek(0)
    return buf

def clean_and_load_markdown_json(md_string):
    try:
        match = re.search(r"```(?:json)?\s*(.*?)```", md_string, re.DOTALL)
        if not match:
            return None
        json_str = match.group(1).strip()
        return json.loads(json_str)
    except Exception:
        return None

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

def extract_model_params(model_path):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    state_dict = torch.load(model_path, map_location=torch.device(device))
    
    vocab_size = state_dict['tok_emb.weight'].size(0)
    block_size = state_dict['pos_emb'].size(1)
    n_embd = state_dict['tok_emb.weight'].size(1)
    n_layer = max([int(key.split('.')[1]) for key in state_dict.keys() if key.startswith('blocks.')]) + 1
    
    n_props = 0
    if 'prop_nn.weight' in state_dict:
        n_props = state_dict['prop_nn.weight'].size(1)
    
    return {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_props': n_props
    }

def writeConn(adj, nodes):
    print('*'*20)
    print(adj, nodes)
    print('*'*20)
    mirror = 1
    
    beam1 = []; beam2 = []
    beam1_1 = []; beam2_1 = []
    beam1_2 = []; beam2_2 = []

    A_plot = adj.copy()
    A_plot[np.triu_indices(A_plot.shape[0])] = 0.
    row = np.nonzero(A_plot)[0]
    col = np.nonzero(A_plot)[1]

    for i, j in zip(row, col):
        beam1_1.append(nodes[i, :])
        beam2_1.append(nodes[j, :])

    if mirror == 1:
        for i, j in zip(beam1_1, beam2_1):
            # -y
            temp1 = i.copy(); temp2 = j.copy()
            temp1[1] = -i[1]; temp2[1] = -j[1]
            beam1_2.append(temp1); beam2_2.append(temp2)
            # -x & -y
            temp1 = i.copy(); temp2 = j.copy()
            temp1[0] = -i[0]; temp2[0] = -j[0]
            temp1[1] = -i[1]; temp2[1] = -j[1]
            beam1_2.append(temp1); beam2_2.append(temp2)
            # -x
            temp1 = i.copy(); temp2 = j.copy()
            temp1[0] = -i[0]; temp2[0] = -j[0]
            beam1_2.append(temp1); beam2_2.append(temp2)

        beam1 = np.concatenate((np.array(beam1_1), np.array(beam1_2)), axis=0)
        beam2 = np.concatenate((np.array(beam2_1), np.array(beam2_2)), axis=0)
    else:
        beam1 = np.array(beam1_1)
        beam2 = np.array(beam2_1)

    # Add the mirrored connections for z-axis
    for i in range(beam1.shape[0]):
        temp1 = beam1[i, :].copy().reshape((1, 3)); temp2 = beam2[i, :].copy().reshape((1, 3))
        temp1[:, -1] = -beam1[i][-1]; temp2[:, -1] = -beam2[i][-1]
        beam1 = np.concatenate((beam1, temp1), axis=0)
        beam2 = np.concatenate((beam2, temp2), axis=0)

    all_nodes = np.concatenate((beam1, beam2))
    
    # Use return_inverse to map original node indices to unique nodes
    uni_nodes, indices = np.unique(all_nodes, axis=0, return_inverse=True)
    
    conn_list = []
    for i in range(beam1.shape[0]):
        n1 = beam1[i]
        n2 = beam2[i]

        # Use np.all(np.isclose()) for accurate node matching
        idx1 = np.where(np.all(np.isclose(uni_nodes, n1), axis=1))[0]
        idx2 = np.where(np.all(np.isclose(uni_nodes, n2), axis=1))[0]
        conn = np.array([idx1, idx2]).squeeze().astype(int)
        conn_list.append(conn)
    
    conn_list = np.array(conn_list).astype(int)
    conn_list = np.unique(conn_list, axis=0)
    
    return uni_nodes, conn_list

# Create a title and a link in the sidebar
with st.sidebar:
    st.markdown("## TrussGPT")
    uploaded_file = st.file_uploader("📤 Upload File", type=["npz", "csv"])
    # Create a slider to select the maximum length, ranging from 0 to 8192, with a default value of 512
    # (Qwen2.5 supports a 128K context and can generate up to 8K tokens)
    # max_length = st.slider("max_length", 0, 8192, 512, step=1)
max_length = 2048


# Create a title and a subtitle
st.title("💬 TrussGPT Chatbot")
st.caption("🚀 LLM-powered inverse design for truss metamaterials")

# Define the model path
mode_name_or_path = '../llms/Qwen2.5-7B-Instruct'

ss_path = '/home/ljx/Ljx/Materials predict/Truss/TrussGPT/model/TrussGPT-Designer/SS.pt'
params = extract_model_params(ss_path)

def extract_model_params(model_path):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(model_path, map_location=torch.device(device)) 
    vocab_size = state_dict['tok_emb.weight'].size(0)
    block_size = state_dict['pos_emb'].size(1)
    n_embd = state_dict['tok_emb.weight'].size(1)
    n_layer = max([int(key.split('.')[1]) for key in state_dict.keys() if key.startswith('blocks.')]) + 1 
    n_props = 0
    if 'prop_nn.weight' in state_dict:
        n_props = state_dict['prop_nn.weight'].size(1)
    return {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_props': n_props
    }

# Define a function to load the model and tokenizer
@st.cache_resource
def get_model():
    # Load tokenizer from the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model from the pre-trained checkpoint and set model parameters
    model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")

    cmodel = cModel()
    cmodel.load_state_dict(torch.load(Truss_Tokenizer_savedFolder+'/best_c_model.pt'))

    Truss_Tokenizer = vaeModel()
    Truss_Tokenizer.load_state_dict(torch.load(Truss_Tokenizer_savedFolder+'/best_model.pt'))
    Truss_Tokenizer.eval()

    mconf2 = GPTConfig(params["vocab_size"], params["block_size"], num_props=61,
                   n_layer=params["n_layer"], n_head=12, n_embd=params["n_embd"],
                   lstm=False, lstm_layers=2)
    TrussGPT_Designer_SS = GPT(mconf2)
    TrussGPT_Designer_SS.load_state_dict(torch.load(ss_path, map_location=torch.device(device)))
    TrussGPT_Designer_SS = TrussGPT_Designer_SS.to(device)
    TrussGPT_Designer_SS.eval()

    return tokenizer, model, Truss_Tokenizer, cmodel, TrussGPT_Designer_SS

def load_gpt(path):
    # print(path)
    params = extract_model_params(path)
    nprops = params["n_props"]
    mconf = GPTConfig(params["vocab_size"], params["block_size"], num_props=nprops,
                   n_layer=params["n_layer"], n_head=12, n_embd=params["n_embd"],
                   lstm=False, lstm_layers=2)
    TrussGPT_Designer = GPT(mconf)
    
    TrussGPT_Designer.load_state_dict(torch.load(path, map_location=torch.device(device)))
    TrussGPT_Designer = TrussGPT_Designer.to(device)
    TrussGPT_Designer.eval()
    return TrussGPT_Designer

tokenizer, model, Truss_Tokenizer, cmodel, TrussGPT_Designer_SS = get_model()

with open('prompt.md', 'r') as f:
    lines = f.readlines()
    prompt = ''.join(lines)
system_prompt = prompt

# print(system_prompt)

# If "messages" is not in session_state, create a list with a default message
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "What can I do for you?"}
    ]

print(system_prompt)

# ====== ✅ If a previously generated structure image exists, display it directly ======
if "generated_image" in st.session_state:
    st.image(st.session_state["generated_image"], caption="Truss Structure", use_container_width=True)

for msg in st.session_state.messages:
    if msg['role'] == 'user':
        with st.chat_message("user", avatar="/home/ljx/Ljx/Materials predict/Truss/TrussGPT/llm_script/web_demo/image/User.png"):  # ← 自定义头像路径
            st.write(msg["content"])
    elif msg['role'] == 'assistant':
        with st.chat_message("assistant", avatar="/home/ljx/Ljx/Materials predict/Truss/TrussGPT/llm_script/web_demo/image/TrussGPT.png"):  # ← 自定义头像路径
            st.write(msg["content"])

# uploaded_file = st.file_uploader("Upload", type=["npz"])

# If the user enters input in the chat box, perform the following actions
if prompt := st.chat_input():
    # Display the user's input in the chat interface
    # st.chat_message("user").write(prompt)
    with st.chat_message("user", avatar="/home/ljx/Ljx/Materials predict/Truss/TrussGPT_Final_v2/llm_script/web_demo/image/User.png"):
        st.write(prompt)
    print(prompt)

    # Append the user's input to the messages list in session_state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send the conversation input to the model and get the response
    input_ids = tokenizer.apply_chat_template(st.session_state.messages,tokenize=False,add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')
    print(model_inputs.input_ids.device)
    print(model.device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=max_length)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    analyse = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(analyse)

    category, result, json_data = parse_analyse(analyse)

    if category == 'predict':
        if uploaded_file is None:
            response = 'Please upload your file'
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:

            # ---------- ① Predict properties ----------
            property_names = ['E11', 'E22', 'E33', 'G23', 'G31', 'G12',
                            'v21', 'v31', 'v32', 'v12', 'v13', 'v23']

            truss_data = np.load(uploaded_file, allow_pickle=True)
            adj_list = torch.tensor(truss_data['adj']).reshape(1, -1).float()
            x = torch.tensor(truss_data['x']).reshape(1, -1).float()

            latent = Truss_Tokenizer.encoder(adj_list, x)[0]
            quantized = Truss_Tokenizer.quantize(latent)[0]
            property_pred = cmodel(quantized)
            property_array = property_pred.cpu().detach().numpy().flatten()

            result_dict = {n: float(round(v, 6)) for n, v in zip(property_names, property_array)}
            result_df   = pd.DataFrame(result_dict.items(), columns=["Property", "Value"])
            # st.dataframe(result_df, use_container_width=True)

            # ---------- ② Construct LLM input ----------
            analysis_prompt = (
                "You are an expert in truss metamaterial design and mechanical performance analysis, "
                "specializing in interpreting elastic properties (such as E11, G23, v21, etc.) "
                "to infer the underlying structural behavior and anisotropic characteristics of the material. "
                "Based on the provided property values, please conduct a thorough and professional analysis from the following four aspects:\n"
                "1. Anisotropy: Analyze differences in elastic moduli (Eij) and Poisson’s ratios (νij) across directions to assess anisotropic behavior.\n"
                "2. Poisson’s Ratio Characteristics: Carefully identify any negative Poisson’s ratios. A negative Poisson’s ratio (ν < 0) means that when the material is stretched in one direction, it also expands (rather than contracts) in the transverse direction—this is known as auxetic behavior. Conversely, a positive Poisson’s ratio (ν > 0) means the material contracts in the transverse direction when stretched. Avoid incorrectly describing negative Poisson’s ratios as 'contraction under tension'. \n"
                "3. Shear Resistance: Evaluate the directional distribution of shear stiffness using Gij values and identify which shear planes are stiffer.\n"
                "4. Compliance: Assess the structure’s ability to conform to external loads by jointly analyzing trends in Poisson’s ratios and elastic moduli. Do not judge compliance based solely on the magnitude of Poisson’s ratios.\n"
                "Your analysis must be technically accurate, logically structured, and consistent with the terminology of engineering mechanics and materials science. "
                "Avoid vague language, formulaic conclusions, and incorrect interpretations of mechanical behavior.\n"
                "Note: The elastic moduli (e.g., Eij and Gij) have been non-dimensionalized and are not expressed in physical units such as GPa.\n"
                f"Here are the input material properties:\n{json.dumps(result_dict, ensure_ascii=False, indent=2)}"
            )

            llm_messages = [
                {"role": "system", "content": "You are an expert large language model dedicated to the field of truss metamaterials."},
                {"role": "user",   "content": analysis_prompt},
            ]

            # ---------- ③ Let the LLM generate the analysis ----------
            llm_input_ids = tokenizer.apply_chat_template(
                llm_messages, tokenize=False, add_generation_prompt=True
            )
            llm_tensors = tokenizer([llm_input_ids], return_tensors="pt").to(model.device)
            llm_out_ids = model.generate(
                llm_tensors.input_ids,
                max_new_tokens=1536,   
                temperature=0.7,
                top_p=0.9
            )
            llm_text = tokenizer.batch_decode(llm_out_ids[:, llm_tensors.input_ids.shape[1]:],
                                            skip_special_tokens=True)[0].strip()

            # ---------- ④ Display everything in the chat window ----------
            response_props = "\n".join([f"{k}: {v:.6f}" for k, v in result_dict.items()])
            combined_resp = (
                f"**Predicted Properties:**\n{response_props}\n\n"
                f"{llm_text}"
            )

            st.session_state.messages.append({"role": "assistant", "content": combined_resp})
            st.markdown(combined_resp)  

    # 2. Inverse design: generate structure based on given properties
    elif category == 'inverse':
        val =  json.loads(result)['val']
        dim = json.loads(result)['dim']
        if dim == ['v21']:
            path = '/home/ljx/Ljx/Materials predict/Truss/TrussGPT/model/TrussGPT-Designer/v21.pt'
        elif dim == ['E33']:
            path = '/home/ljx/Ljx/Materials predict/Truss/TrussGPT/model/TrussGPT-Designer/E33.pt'
        elif dim == ['E33', 'v21']:
            path = '/home/ljx/Ljx/Materials predict/Truss/TrussGPT/model/TrussGPT-Designer/E33v21.pt'
        elif dim == ['E11', 'E22', 'E33']:
            path = '/home/ljx/Ljx/Materials predict/Truss/TrussGPT/model/TrussGPT-Designer/E11E22E33.pt'
        elif dim == ['E11', 'E22', 'E33', 'v21']:
            path = '/home/ljx/Ljx/Materials predict/Truss/TrussGPT/model/TrussGPT-Designer/E11E22E33v21.pt'
        else:
            val = [0.1]
            path = '/home/ljx/Ljx/Materials predict/Truss/TrussGPT/model/TrussGPT-Designer/v21.pt'

        TrussGPT_Designer = load_gpt(path)
        nprops = len(val)
        c = val
        x = torch.tensor([256], dtype=torch.long)[None,...].repeat(1, 1).to(device)
        p = torch.tensor([c], dtype=torch.float32).repeat(1, 1)
        if nprops > 1:
            p = p.unsqueeze(1)
        p = p.to(device)
        print(p.device, x.device)
        y = TrussGPT_Designer.sample(x, params["block_size"], temperature=1.2, do_sample=True, top_k=0., top_p=0.9, prop=p)
        props_str = ",".join([str(i) for i in c])
        for gen_mol in y:
            completion = ",".join([str(int(i)) for i in gen_mol[:16]])
            if '256' in completion or '257' in completion:
                print('Invalid output format  ：'+completion)
            else:
                truss_str = props_str + "," + completion
        completion = torch.tensor([int(x) for x in completion.split(",")]).view(1, -1)
        
        adj_decoded, x_decoded = Truss_Tokenizer.decoder(Truss_Tokenizer.recover(completion))
        threshold = 0.6
        adj_binary = (adj_decoded >= threshold).float().cpu().numpy().reshape(-1,15,15)  
        
        x_decoded = np.genfromtxt('/home/ljx/Ljx/Materials predict/Truss/TrussGPT/llm_script/web_demo/nodesInit.csv', delimiter=",")

        for i in range(adj_binary.shape[0]):
            adj = adj_binary[i]
            adj = adj + adj.T  
            np.fill_diagonal(adj, 0)  
            adj[adj != 0] = 1

            uni_nodes, conn_list = writeConn(adj, x_decoded)

            octant = {
                    'adj': adj,
                    'x': x_decoded,
                    }
            output_filename_octant = 'truss/octant_truss_{}.npz'.format(i)

            np.savez(output_filename_octant, **octant)

            structure_info = {
                    'uni_nodes': uni_nodes,
                    'conn_list': conn_list,
                    }

            output_filename = 'truss/truss_{}.npz'.format(i)

            np.savez(output_filename, **structure_info)
        
        with open("truss/octant_truss_0.npz", "rb") as f:
            st.download_button(
                label="📥 Download Octant Structure",
                data=f,
                file_name="octant.npz",
                mime="application/octet-stream"
            )
        
        with open("truss/truss_0.npz", "rb") as f:
            st.download_button(
                label="📥 Download Complete Structure",
                data=f,
                file_name="truss.npz",
                mime="application/octet-stream"
            )
            
        data = np.load('truss/truss_0.npz')
        nodes = data['uni_nodes']
        edges = data['conn_list']

        # Generate and display the image
        image_buf = plot_truss_structure_transparent(nodes, edges)
        st.session_state["generated_image"] = image_buf  
        st.image(image_buf, caption="Truss Structure", use_container_width=True)
        response = "Design successfully generated. Feel free to download it."
        st.session_state.messages.append({"role": "assistant", "content": response})
        
    # 3. generate structure based on the given curve
    elif category == 'ss':
        if uploaded_file == None:
            response = 'Please upload your file to get started.'
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            c = pd.read_csv(uploaded_file, header=None).iloc[0].tolist()
            nprops = 61
            x = torch.tensor([256], dtype=torch.long)[None,...].repeat(1, 1).to(device)
            p = torch.tensor([c]).repeat(1, 1).to(device) if nprops == 1 else torch.tensor([c]).repeat(1, 1).unsqueeze(1).to(device)
            print(p.device, x.device)
            y = TrussGPT_Designer_SS.sample(x, params["block_size"], temperature=1.2, do_sample=True, top_k=0., top_p=0.9, prop=p)
            props_str = ",".join([str(i) for i in c])
            for gen_mol in y:
                completion = ",".join([str(int(i)) for i in gen_mol[:16]])
                if '256' in completion or '257' in completion:
                    print('Invalid output format  ：'+completion)
                else:
                    truss_str = props_str + "," + completion
            completion = torch.tensor([int(com) for com in completion.split(",")]).view(1, -1)
            
            adj_decoded, x_decoded = Truss_Tokenizer.decoder(Truss_Tokenizer.recover(completion))
            print(adj_decoded.shape, x_decoded.shape)
            threshold = 0.6
            adj_binary = (adj_decoded >= threshold).float().cpu().numpy().reshape(-1,15,15) 
            x_decoded = np.genfromtxt('/home/ljx/Ljx/Materials predict/Truss/Truss_Generation_v2/nodesInit.csv', delimiter=",")    
            for i in range(adj_binary.shape[0]):
                adj = adj_binary[i]
                adj = adj + adj.T  
                np.fill_diagonal(adj, 0) 
                adj[adj != 0] = 1

                uni_nodes, conn_list = writeConn(adj, x_decoded)

                octant = {
                    'adj': adj,
                    'x': x_decoded,
                    }
                    
                output_filename_octant = 'truss/octant_truss_{}.npz'.format(i)

                np.savez(output_filename_octant, **octant)

                structure_info = {
                        'uni_nodes': uni_nodes,
                        'conn_list': conn_list,
                        }

                output_filename = 'truss/truss_{}.npz'.format(i)

                np.savez(output_filename, **structure_info)
            
            with open("truss/octant_truss_0.npz", "rb") as f:
                st.download_button(
                    label="📥 Download Octant Structure",
                    data=f,
                    file_name="octant.npz",
                    mime="application/octet-stream"
                )
            
            with open("truss/truss_0.npz", "rb") as f:
                st.download_button(
                    label="📥 Download Complete Structure",
                    data=f,
                    file_name="truss.npz",
                    mime="application/octet-stream"
                )
            data = np.load('truss/truss_0.npz')
            nodes = data['uni_nodes']
            edges = data['conn_list']

            image_buf = plot_truss_structure_transparent(nodes, edges)
            st.session_state["generated_image"] = image_buf  
            st.image(image_buf, caption="Truss Structure", use_container_width=True)
            
            response = "The truss structure has been successfully designed based on your input. Please download the file to view or simulate."
            st.session_state.messages.append({"role": "assistant", "content": response})
        
    else:
        response = result
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Display the model's output in the chat interface
    
    if 'response' in locals():
        with st.chat_message("assistant", avatar="/home/ljx/Ljx/Materials predict/Truss/TrussGPT_Final_v2/llm_script/web_demo/image/TrussGPT.png"):
            st.write(response)

