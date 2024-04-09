import time

from models_settings import get_model_metadata, update_model_parameters
from models import load_model
import shared
from text_generation import start_chat, encode

from flask import Flask, request, Response, stream_with_context
from queue import Queue
from threading import Lock, Timer
import threading

# 创建一个 Flask 应用程序
app = Flask(__name__)

class State:
    def __init__(self):
        self.stop_everything = False

class Task:
    def __init__(self, user_input, stream):
        self.user_input = user_input
        self.priority = None
        self.stream = stream
        self.need_save_content = False  #需要保存中间值
        self.is_done = False  # 是否完成任务
        self.is_preempt = False  # 是否发生了抢占
        self.pre_output = ''
        self.cache = ''
        self.state = State()

    def done(self):
        self.is_done = True

    def stop(self):
        print('stop')
        self.state.stop_everything = True

    def is_stop(self):
        return self.state.stop_everything


class MLFQ:
    cur_task: Task | None
    def __init__(self, num_queues, time_slice):
        self.num_queues = num_queues
        self.queues = [Queue() for _ in range(num_queues)]
        self.time_slice = time_slice
        self.lock = Lock()
        self.lock_timer = None  # 锁的计时器
        self.cur_task = None

    def add_task(self, task):
        tokens_len = len(encode(task.user_input)[0])
        priority = 0 if tokens_len <= 128 else 1 if 128 < tokens_len <= 256 else 2
        self.queues[priority].put(task)
        task.priority = priority

    def get_highest_priority_queue(self):
        for i, queue in enumerate(self.queues):
            if not queue.empty():
                return i, queue.get()
        return None, None

    def has_higher_priority_task(self, current_priority):
        for i, queue in enumerate(self.queues):
            if not queue.empty() and i < current_priority:
                return True
        return False

    def save_content_no_preempt(self, task):
        task.need_save_content = True
        self.stop_task()

    def degraded(self, task):
        if task.priority == self.num_queues-1:
            self.queues[task.priority].put(task)  # 已经是最低优先级：放回队尾，不降级
        else:
            self.queues[task.priority+1].put(task)  # 降级处理

    def run_task(self, task: Task):
        self.cur_task = task
        self.cur_task.state.stop_everything = False
        for value in start_chat(task, shared.state, shared.stopping_strings, True):
            temp: str = value['visible'][-1][1]
            task.cache = task.cache + temp.replace(task.pre_output, '', 1)
            task.pre_output = temp
        if not self.cur_task.is_stop():
            task.done()
            self.release()

    def release(self):
        if self.lock_timer:
            self.lock_timer.cancel()
        self.lock.release()

    def save_content_preempt(self):
        self.cur_task.need_save_content = True
        self.cur_task.is_preempt = True
        self.stop_task()

    def stop_task(self):
        # 暂停当前任务
        self.cur_task.stop()
        if (self.cur_task.is_preempt):
            # 当前任务未执行完毕，放回队列
            self.queues[self.cur_task.priority].put(self.cur_task)
        else:
            self.degraded(self.cur_task)
            # 释放锁，自动执行下一个任务
            self.release()

    def run(self):
        while True:
            if not self.lock.locked():
                priority, task = mlfq.get_highest_priority_queue()
                # 无新的 task
                if task is None:
                    time.sleep(1)
                # 有新 task + 有空执行
                elif task is not None:
                    self.lock.acquire()
                    # 启动锁的计时器
                    self.lock_timer = Timer(self.time_slice[priority], lambda: self.save_content_no_preempt(task))
                    self.lock_timer.start()
                    start_time = time.time()
                    # 执行当前任务
                    self.run_task(task)
                    # task.stream = start_chat(task.user_input, shared.state, shared.stopping_strings, True)
                    # 检查锁在locked情况下，是否有更高优先级任务到来
                    while self.lock.locked():
                        if self.has_higher_priority_task(priority):
                            # 已执行时间
                            done_time = time.time() - start_time
                            # 执行完最小时间片，直接抢占
                            if done_time >= min_time_slice:
                                self.save_content_preempt()
                            else:
                                remaining_time = min_time_slice - done_time
                                self.lock_timer.cancel()  # 取消计时器
                                # 启动锁的计时器
                                self.lock_timer = Timer(remaining_time, lambda: self.save_content_preempt())
                                self.lock_timer.start()


# 生成event-stream
def generate(task: Task):
    while True:
        if not task.is_done or task.cache:
            data = task.cache.replace("\n", ' ', -1)
            yield f'data: Event: {data}\n\n'
            task.cache = ''
        else:
            yield '\n\n'
            break
        time.sleep(1)

# 加载模型：默认llama2-7b 4bit量化版本
def load_llama_model():
    global model_loaded
    model_name = 'llama2-7b-ggml-model-q4_0.gguf'
    model_settings = get_model_metadata(model_name)
    update_model_parameters(model_settings, initial=True)  # hijack the command-line arguments
    # Load the model
    shared.model, shared.tokenizer = load_model(model_name, 'llama.cpp')
    model_loaded = True


# 定义一个路由，指定 URL 路径和 HTTP 方法
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    # 如果模型未加载，则先加载模型
    if not model_loaded:
        load_llama_model()
    data = request.json
    if 'textbox' in data:
        user_input = data['textbox']
        if user_input.strip() != '':
            # 创建任务对象
            task = Task(user_input, None)
            mlfq.add_task(task)
            return Response(stream_with_context(generate(task)), content_type='text/event-stream')
        else:
            return Response('请输入文本')
    else:
        return Response('error')


# 在主进程中加载模型
if __name__ == '__main__':
    # 初始化模型加载状态为未加载
    model_loaded = False
    # 预先加载模型
    load_llama_model()

    # 创建一个 MLFQ 实例
    num_queues = 3
    min_time_slice = 10
    time_slice = [min_time_slice, min_time_slice * 2, min_time_slice * 3]
    mlfq = MLFQ(num_queues, time_slice)

    # 创建一个线程来运行 MLFQ
    mlfq_thread = threading.Thread(target=mlfq.run)
    mlfq_thread.start()

    # 在主线程中运行 Flask 应用程序
    app.run(host='localhost', port=8080, debug=False)
