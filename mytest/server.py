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


class Task:
    def __init__(self, user_input, stream):
        self.user_input = user_input
        self.priority = None
        self.stream = stream
        self.need_save_content = False  # 需要保存中间值
        self.is_done = False  # 是否完成任务
        self.is_preempt = False  # 是否发生了抢占


class MLFQ:
    def __init__(self, num_queues, time_slice):
        self.num_queues = num_queues
        self.queues = [Queue() for _ in range(num_queues)]
        self.time_slice = time_slice
        self.lock = Lock()
        self.lock_timer = None  # 锁的计时器

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

    def save_content_preempt(self, task):
        task.need_save_content = True
        task.is_preempt = True

    def degraded(self, task):
        if task.priority == self.num_queues-1:
            self.queues[task.priority].put(task)  # 已经是最低优先级：放回队尾，不降级
        else:
            self.queues[task.priority+1].put(task)  # 降级处理

    def run(self):
        while True:
            if not self.lock.locked():
                priority, task = mlfq.get_highest_priority_queue()
                # 无新的 task
                if task is None:
                    time.sleep(1)
                # 有新 task + 有空执行
                elif task is not None:
                    print("locked")
                    self.lock.acquire()
                    # 启动锁的计时器
                    self.lock_timer = Timer(self.time_slice[priority], lambda: self.save_content_no_preempt(task))
                    self.lock_timer.start()
                    start_time = time.time()
                    # 执行当前任务
                    task.stream = start_chat(task.user_input, shared.state, shared.stopping_strings, True)
                    # 检查锁在locked情况下，是否有更高优先级任务到来
                    while self.lock.locked():
                        if self.has_higher_priority_task(priority):
                            # 已执行时间
                            done_time = time.time() - start_time
                            # 执行完最小时间片，直接抢占
                            if done_time >= min_time_slice:
                                self.lock_timer.cancel()  # 取消计时器
                                self.save_content_preempt(task)
                            else:
                                remaining_time = min_time_slice - done_time
                                self.lock_timer.cancel()  # 取消计时器
                                # 启动锁的计时器
                                self.lock_timer = Timer(self.time_slice[priority], lambda: self.save_content_preempt(task))
                                self.lock_timer.start()


# 生成event-stream
def generate(task, mlfq):
    try:
        for value in task.stream:
            if task.need_save_content:
                task.user_input = task.user_input + value['visible'][-1][1]  # 保存内容
                task.stream = None
                if task.is_preempt:
                    mlfq.queues[task.priority].put(task)
                else:
                    mlfq.degraded(task)  # 降级处理
                break
            yield f"data: {value['visible'][-1][1]}\n\n"
        yield "\n"
        if not task.need_save_content:
            task.is_done = True
        task.need_save_content = False
    finally:
        # 在生成器完成时:清空计时器 + 释放锁
        if mlfq.lock_timer:
            mlfq.lock_timer.cancel()
        mlfq.lock.release()
        print("released")


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
            print("add")
            while True:
                if task.stream is not None:
                    generate(task, mlfq)
                if task.is_done:
                    break
            return ''
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
