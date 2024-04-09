import json
import re
from pathlib import Path

from transformers import is_torch_xpu_available
import torch

import chat
import metadata_gguf
import shared


def get_model_metadata(model):
    model_settings = {}

    # Get settings from models/config.yaml and models/config-user.yaml
    settings = shared.model_config
    # print(f'settings={settings}')
    for pat in settings:
        if re.match(pat.lower(), model.lower()):
            for k in settings[pat]:
                model_settings[k] = settings[pat][k]

    path = Path(f'models/{model}/config.json')
    # print(f'path={path}')
    if path.exists():
        hf_metadata = json.loads(open(path, 'r', encoding='utf-8').read())
    else:
        hf_metadata = None

    if 'loader' not in model_settings:
        if hf_metadata is not None and 'quip_params' in hf_metadata:
            loader = 'QuIP#'
        else:
            loader = infer_loader(model, model_settings)

        model_settings['loader'] = loader

    # GGUF metadata
    if model_settings['loader'] in ['llama.cpp', 'llamacpp_HF', 'ctransformers']:
        path = Path(f'{shared.args.model_dir}/{model}')
        if path.is_file():
            model_file = path
        else:
            model_file = list(path.glob('*.gguf'))[0]

        metadata = metadata_gguf.load_metadata(model_file)
        if 'llama.context_length' in metadata:
            model_settings['n_ctx'] = metadata['llama.context_length']
        if 'llama.rope.scale_linear' in metadata:
            model_settings['compress_pos_emb'] = metadata['llama.rope.scale_linear']
        if 'llama.rope.freq_base' in metadata:
            model_settings['rope_freq_base'] = metadata['llama.rope.freq_base']
        if 'tokenizer.chat_template' in metadata:
            template = metadata['tokenizer.chat_template']
            eos_token = metadata['tokenizer.ggml.tokens'][metadata['tokenizer.ggml.eos_token_id']]
            bos_token = metadata['tokenizer.ggml.tokens'][metadata['tokenizer.ggml.bos_token_id']]
            template = template.replace('eos_token', "'{}'".format(eos_token))
            template = template.replace('bos_token', "'{}'".format(bos_token))

            template = re.sub(r'raise_exception\([^)]*\)', "''", template)
            model_settings['instruction_template'] = 'Custom (obtained from model metadata)'
            model_settings['instruction_template_str'] = template

    else:
        # Transformers metadata
        if hf_metadata is not None:
            metadata = json.loads(open(path, 'r', encoding='utf-8').read())
            if 'max_position_embeddings' in metadata:
                model_settings['truncation_length'] = metadata['max_position_embeddings']
                model_settings['max_seq_len'] = metadata['max_position_embeddings']

            if 'rope_theta' in metadata:
                model_settings['rope_freq_base'] = metadata['rope_theta']

            if 'rope_scaling' in metadata and type(metadata['rope_scaling']) is dict and all(
                    key in metadata['rope_scaling'] for key in ('type', 'factor')):
                if metadata['rope_scaling']['type'] == 'linear':
                    model_settings['compress_pos_emb'] = metadata['rope_scaling']['factor']

            if 'quantization_config' in metadata:
                if 'bits' in metadata['quantization_config']:
                    model_settings['wbits'] = metadata['quantization_config']['bits']
                if 'group_size' in metadata['quantization_config']:
                    model_settings['groupsize'] = metadata['quantization_config']['group_size']
                if 'desc_act' in metadata['quantization_config']:
                    model_settings['desc_act'] = metadata['quantization_config']['desc_act']

        # Read AutoGPTQ metadata
        path = Path(f'{shared.args.model_dir}/{model}/quantize_config.json')
        if path.exists():
            metadata = json.loads(open(path, 'r', encoding='utf-8').read())
            if 'bits' in metadata:
                model_settings['wbits'] = metadata['bits']
            if 'group_size' in metadata:
                model_settings['groupsize'] = metadata['group_size']
            if 'desc_act' in metadata:
                model_settings['desc_act'] = metadata['desc_act']

    # Try to find the Jinja instruct template
    path = Path(f'models/{model}') / 'tokenizer_config.json'
    if path.exists():
        metadata = json.loads(open(path, 'r', encoding='utf-8').read())
        if 'chat_template' in metadata:
            template = metadata['chat_template']
            for k in ['eos_token', 'bos_token']:
                if k in metadata:
                    value = metadata[k]
                    if type(value) is dict:
                        value = value['content']

                    template = template.replace(k, "'{}'".format(value))

            template = re.sub(r'raise_exception\([^)]*\)', "''", template)
            model_settings['instruction_template'] = 'Custom (obtained from model metadata)'
            model_settings['instruction_template_str'] = template

    if 'instruction_template' not in model_settings:
        model_settings['instruction_template'] = 'Alpaca'

    if model_settings['instruction_template'] != 'Custom (obtained from model metadata)':
        model_settings['instruction_template_str'] = chat.load_instruction_template(
            model_settings['instruction_template'])

    # Ignore rope_freq_base if set to the default value
    if 'rope_freq_base' in model_settings and model_settings['rope_freq_base'] == 10000:
        model_settings.pop('rope_freq_base')

    # Apply user settings from models/config-user.yaml
    settings = shared.user_config
    for pat in settings:
        if re.match(pat.lower(), model.lower()):
            for k in settings[pat]:
                model_settings[k] = settings[pat][k]

    return model_settings


def update_model_parameters(state, initial=False):
    '''
    UI: update the command-line arguments based on the interface values
    '''
    elements = list_model_elements()  # the names of the parameters
    gpu_memories = []

    for i, element in enumerate(elements):
        if element not in state:
            continue

        value = state[element]
        if element.startswith('gpu_memory'):
            gpu_memories.append(value)
            continue

        if initial and element in shared.provided_arguments:
            continue

        # Setting null defaults
        if element in ['wbits', 'groupsize', 'model_type'] and value == 'None':
            value = vars(shared.args_defaults)[element]
        elif element in ['cpu_memory'] and value == 0:
            value = vars(shared.args_defaults)[element]

        # Making some simple conversions
        if element in ['wbits', 'groupsize', 'pre_layer']:
            value = int(value)
        elif element == 'cpu_memory' and value is not None:
            value = f"{value}MiB"

        if element in ['pre_layer']:
            value = [value] if value > 0 else None

        setattr(shared.args, element, value)

    found_positive = False
    for i in gpu_memories:
        if i > 0:
            found_positive = True
            break

    if not (initial and vars(shared.args)['gpu_memory'] != vars(shared.args_defaults)['gpu_memory']):
        if found_positive:
            shared.args.gpu_memory = [f"{i}MiB" for i in gpu_memories]
        else:
            shared.args.gpu_memory = None


def infer_loader(model_name, model_settings):
    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    if not path_to_model.exists():
        loader = None
    elif (path_to_model / 'quantize_config.json').exists() or (
            'wbits' in model_settings and type(model_settings['wbits']) is int and model_settings['wbits'] > 0):
        loader = 'ExLlamav2_HF'
    elif (path_to_model / 'quant_config.json').exists() or re.match(r'.*-awq', model_name.lower()):
        loader = 'AutoAWQ'
    elif len(list(path_to_model.glob('*.gguf'))) > 0 and path_to_model.is_dir() and (
            path_to_model / 'tokenizer_config.json').exists():
        loader = 'llamacpp_HF'
    elif len(list(path_to_model.glob('*.gguf'))) > 0:
        loader = 'llama.cpp'
    elif re.match(r'.*\.gguf', model_name.lower()):
        loader = 'llama.cpp'
    elif re.match(r'.*exl2', model_name.lower()):
        loader = 'ExLlamav2_HF'
    elif re.match(r'.*-hqq', model_name.lower()):
        return 'HQQ'
    else:
        loader = 'Transformers'

    return loader


def list_model_elements():
    elements = [
        'loader',
        'filter_by_loader',
        'cpu_memory',
        'auto_devices',
        'disk',
        'cpu',
        'bf16',
        'load_in_8bit',
        'trust_remote_code',
        'no_use_fast',
        'use_flash_attention_2',
        'load_in_4bit',
        'compute_dtype',
        'quant_type',
        'use_double_quant',
        'wbits',
        'groupsize',
        'model_type',
        'pre_layer',
        'triton',
        'desc_act',
        'no_inject_fused_attention',
        'no_inject_fused_mlp',
        'no_use_cuda_fp16',
        'disable_exllama',
        'disable_exllamav2',
        'cfg_cache',
        'no_flash_attn',
        'num_experts_per_token',
        'cache_8bit',
        'cache_4bit',
        'autosplit',
        'threads',
        'threads_batch',
        'n_batch',
        'no_mmap',
        'mlock',
        'no_mul_mat_q',
        'n_gpu_layers',
        'tensor_split',
        'n_ctx',
        'gpu_split',
        'max_seq_len',
        'compress_pos_emb',
        'alpha_value',
        'rope_freq_base',
        'numa',
        'logits_all',
        'no_offload_kqv',
        'row_split',
        'tensorcores',
        'hqq_backend',
    ]
    if is_torch_xpu_available():
        for i in range(torch.xpu.device_count()):
            elements.append(f'gpu_memory_{i}')
    else:
        for i in range(torch.cuda.device_count()):
            elements.append(f'gpu_memory_{i}')

    return elements
