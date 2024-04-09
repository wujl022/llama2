import time
from pathlib import Path


import transformers

import shared
import sampler_hijack
from logging_colors import logger
from models_settings import get_model_metadata
from llamacpp_model import LlamaCppModel

transformers.logging.set_verbosity_error()
sampler_hijack.hijack_samplers()


def load_model(model_name, loader=None):
    logger.info(f"Loading \"{model_name}\"")
    t0 = time.time()

    shared.is_seq2seq = False
    shared.model_name = model_name
    metadata = get_model_metadata(model_name)
    if loader is None:
        if shared.args.loader is not None:
            loader = shared.args.loader
        else:
            loader = metadata['loader']
            if loader is None:
                logger.error('The path to the model does not exist. Exiting.')
                raise ValueError

    shared.args.loader = loader
    output = llamacpp_loader(model_name)
    if type(output) is tuple:
        model, tokenizer = output
    else:
        model = output
        if model is None:
            return None, None
        else:
            tokenizer = load_tokenizer(model_name, model)

    shared.settings.update({k: v for k, v in metadata.items() if k in shared.settings})
    if loader.lower().startswith('exllama'):
        shared.settings['truncation_length'] = shared.args.max_seq_len
    elif loader in ['llama.cpp', 'llamacpp_HF', 'ctransformers']:
        shared.settings['truncation_length'] = shared.args.n_ctx

    logger.info(f"LOADER: \"{loader}\"")
    logger.info(f"TRUNCATION LENGTH: {shared.settings['truncation_length']}")
    logger.info(f"INSTRUCTION TEMPLATE: \"{metadata['instruction_template']}\"")
    t1 = time.time()
    logger.info(f"Loaded the model in {(t1 - t0):.2f} seconds.")

    # csv_file_path = './performance/change_model.csv'
    # with open(csv_file_path, mode='a', newline='') as change_model_file:
    #     change_model_writer = csv.writer(change_model_file)
    #     # if not change_model_file.tell():
    #     #     change_model_writer.writerow(['model_name', 'seconds'])
    #     change_model_writer.writerow([model_name, round(t1-t0, 2)])
    #     change_model_file.close()
    return model, tokenizer


def llamacpp_loader(model_name):
    path = Path(f'/home/wujiali/LLM/text-generation-webui/models/{model_name}')
    if path.is_file():
        model_file = path
    else:
        model_file = list(Path(f'{shared.args.model_dir}/{model_name}').glob('*.gguf'))[0]

    logger.info(f"llama.cpp weights detected: \"{model_file}\"")
    model, tokenizer = LlamaCppModel.from_pretrained(model_file)
    return model, tokenizer


def load_tokenizer(model_name, model):
    tokenizer = None
    path_to_model = Path(f"{shared.args.model_dir}/{model_name}/")
    if any(s in model_name.lower() for s in ['gpt-4chan', 'gpt4chan']) and Path(
            f"{shared.args.model_dir}/gpt-j-6B/").exists():
        tokenizer = transformers.AutoTokenizer.from_pretrained(Path(f"{shared.args.model_dir}/gpt-j-6B/"))
    elif path_to_model.exists():
        if shared.args.no_use_fast:
            logger.info('Loading the tokenizer with use_fast=False.')

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            path_to_model,
            trust_remote_code=shared.args.trust_remote_code,
            use_fast=not shared.args.no_use_fast
        )

    return tokenizer
