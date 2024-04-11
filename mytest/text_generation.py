import time
import traceback
import random
import ast
import copy
import html
import gc
import re
import json

import torch
import numpy as np
from transformers import is_torch_xpu_available
from accelerate.utils import is_xpu_available

import shared
from extensions import apply_extensions
from logging_colors import logger


local_rank = None


def generate_reply(*args, **kwargs):
    # shared.generation_lock.acquire()
    try:
        for result in _generate_reply(*args, **kwargs):
            yield result
    except Exception as e:
        print(f'error={e}')
    # finally:
        # shared.generation_lock.release()


def _generate_reply(question, state, stopping_strings=None, is_chat=False, escape_html=False, for_ui=False):
    # print(f'question={question}')
    # question = prompt
    # Find the appropriate generation function
    generate_func = apply_extensions('custom_generate_reply')
    if generate_func is None:
        if shared.model_name == 'None' or shared.model is None:
            logger.error("No model is loaded! Select one in the Model tab.")
            yield ''
            return

        if shared.model.__class__.__name__ in ['LlamaCppModel', 'Exllamav2Model', 'CtransformersModel']:
            generate_func = generate_reply_custom
        # print(f'generate_func2={generate_func}')

    # Prepare the input
    original_question = question
    if not is_chat:
        state = apply_extensions('state', state)
        question = apply_extensions('input', question, state)

    # Find the stopping strings
    all_stop_strings = []
    # print(f'stopping_strings={stopping_strings}')
    # print(f"custom_stopping_strings={state['custom_stopping_strings']}")
    for st in (stopping_strings, state['custom_stopping_strings']):
        if type(st) is str:
            # print("2")
            st = ast.literal_eval(f"[{st}]")
            # print(f'st={st}')

        if type(st) is list and len(st) > 0:
            # print("3")
            all_stop_strings += st

    shared.stop_everything = False
    clear_torch_cache()
    seed = set_manual_seed(state['seed'])
    last_update = -1
    reply = ''
    is_stream = state['stream']
    # print(f'is_stream={is_stream}')
    # print(f'all_stop_strings={all_stop_strings}')
    if len(all_stop_strings) > 0 and not state['stream']:
        print("1")
        state = copy.deepcopy(state)
        state['stream'] = True

    min_update_interval = 0
    if state.get('max_updates_second', 0) > 0:
        min_update_interval = 1 / state['max_updates_second']

    # Generate

    # print(f'question={question}')
    # print(f'original_question={original_question}')
    # print(f'seed={seed}')
    # print(f'state={state}')
    # print(f'generate_func()={generate_func(question, original_question, seed, state, stopping_strings, is_chat=is_chat)}')
    for reply in generate_func(question, original_question, seed, state, stopping_strings, is_chat=is_chat):
        reply, stop_found = apply_stopping_strings(reply, all_stop_strings)
        # print(f'reply={reply}')
        # print(f'stop_found={stop_found}')
        if escape_html:
            reply = html.escape(reply)
        if is_stream:
            cur_time = time.time()

            # Maximum number of tokens/second
            # print(f"state['max_tokens_second']={state['max_tokens_second']}")
            if state['max_tokens_second'] > 0:
                diff = 1 / state['max_tokens_second'] - (cur_time - last_update)
                # print(f'diff={diff}')
                if diff > 0:
                    time.sleep(diff)

                last_update = time.time()
                yield reply

            # Limit updates to avoid lag in the Gradio UI
            # API updates are not limited
            else:
                if cur_time - last_update > min_update_interval:
                    # print("here")
                    last_update = cur_time
                    yield reply

        if stop_found or (state['max_tokens_second'] > 0 and shared.stop_everything):
            break

    if not is_chat:
        # print("there")
        reply = apply_extensions('output', reply, state)

    yield reply


def apply_stopping_strings(reply, all_stop_strings):
    stop_found = False
    for string in all_stop_strings:
        idx = reply.find(string)
        if idx != -1:
            reply = reply[:idx]
            stop_found = True
            break

    if not stop_found:
        # If something like "\nYo" is generated just before "\nYou:"
        # is completed, trim it
        for string in all_stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:
                    reply = reply[:-j]
                    break
            else:
                continue

            break

    return reply, stop_found


def set_manual_seed(seed):
    seed = int(seed)
    if seed == -1:
        seed = random.randint(1, 2 ** 31)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif is_torch_xpu_available():
        torch.xpu.manual_seed_all(seed)

    return seed


def encode(prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
    if shared.tokenizer is None:
        raise ValueError('No tokenizer is loaded')

    if shared.model.__class__.__name__ in ['LlamaCppModel', 'CtransformersModel', 'Exllamav2Model']:
        input_ids = shared.tokenizer.encode(str(prompt))
        if shared.model.__class__.__name__ not in ['Exllamav2Model']:
            input_ids = np.array(input_ids).reshape(1, len(input_ids))
    else:
        input_ids = shared.tokenizer.encode(str(prompt), return_tensors='pt', add_special_tokens=add_special_tokens)
        if not add_bos_token:
            while len(input_ids[0]) > 0 and input_ids[0][0] == shared.tokenizer.bos_token_id:
                input_ids = input_ids[:, 1:]

    # Handling truncation
    if truncation_length is not None:
        input_ids = input_ids[:, -truncation_length:]

    if shared.model.__class__.__name__ in ['LlamaCppModel', 'Exllamav2Model', 'CtransformersModel'] or shared.args.cpu:
        return input_ids
    elif shared.args.deepspeed:
        return input_ids.to(device=local_rank)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        return input_ids.to(device)
    elif is_torch_xpu_available():
        return input_ids.to("xpu:0")
    else:
        return input_ids.cuda()


def generate_reply_custom(question, original_question, seed, state, stopping_strings=None, is_chat=False):
    """
    For models that do not use the transformers library for sampling
    """
    seed = set_manual_seed(state['seed'])

    t0 = time.time()
    reply = ''
    try:
        if not is_chat:
            yield ''

        if not state['stream']:
            reply = shared.model.generate(question, state)
            yield reply
        else:
            for reply in shared.model.generate_with_streaming(question, state):
                yield reply

    except Exception:
        traceback.print_exc()
    finally:
        t1 = time.time()
        original_tokens = len(encode(original_question)[0])
        new_tokens = len(encode(original_question + reply)[0]) - original_tokens
        # seconds=t1-t0;
        # tokens/s=new_tokens/(t1-t0);
        # tokens=new_tokens;
        # context=original_tokens;

        # CSV文件路径
        # csv_file_path = './performance/get_tokens.csv'
        # 确保CSV文件存在，并且有一个标题行
        # with open(csv_file_path, mode='a', newline='') as get_tokens_file:
        #     get_tokens_writer = csv.writer(get_tokens_file)
        #     # 如果文件是空的，添加标题行
        #     # if not get_tokens_file.tell():
        #     #     get_tokens_writer.writerow(['time', 'seconds', 'tokens/s', 'tokens', 'context'])
        #     get_tokens_writer.writerow(([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), round(t1-t0, 2), round(new_tokens/(t1-t0), 2), new_tokens, original_tokens]))
        #     get_tokens_file.close()

        print(
            f'Output generated in {(t1 - t0):.2f} seconds ({new_tokens / (t1 - t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
        return


def get_max_prompt_length(state):
    return state['truncation_length'] - state['max_new_tokens']


def clear_torch_cache():
    gc.collect()
    if not shared.args.cpu:
        if is_xpu_available():
            torch.xpu.empty_cache()
        else:
            torch.cuda.empty_cache()


def start_chat(task, state, stopping_strings, for_ui):
    text = task.user_input
    history = state['history']
    output = copy.deepcopy(history)
    output = apply_extensions('history', output)
    state = apply_extensions('state', state)
    is_stream = state['stream']
    # Prepare the input
    visible_text = html.escape(text)
    # Apply extensions
    text, visible_text = apply_extensions('chat_input', text, visible_text, state)
    text = apply_extensions('input', text, state, is_chat=True)
    output['internal'].append([text, ''])
    output['visible'].append([visible_text, ''])

    # Generate the prompt
    prompt = shared.prompt + text + f'\nAI: {task.pre_output}'
    for j, reply in enumerate(generate_reply(prompt, state, stopping_strings, True, for_ui)):

        # Extract the reply
        visible_reply = reply
        if state['mode'] in ['chat', 'chat-instruct']:
            visible_reply = re.sub("(<USER>|<user>|{{user}})", state['name1'], reply)

        visible_reply = html.escape(visible_reply)

        if task.state.stop_everything:
            output['visible'][-1][1] = apply_extensions('output', output['visible'][-1][1], state, is_chat=True)
            yield output
            return
        elif not (j == 0 and visible_reply.strip() == ''):
            output['internal'][-1] = [text, reply.lstrip(' ')]
            output['visible'][-1] = [visible_text, visible_reply.lstrip(' ')]
            if is_stream:
                # print("1")
                yield output

    output['visible'][-1][1] = apply_extensions('output', output['visible'][-1][1], state, is_chat=True)
    # 更新state
    # print('state', state)
    # json_state = json.loads(state)
    # json_state['history'] = output
    # shared.state = json.dumps(json_state)
    return 0