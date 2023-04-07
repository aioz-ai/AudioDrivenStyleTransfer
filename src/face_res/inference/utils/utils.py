"""
aioz.aiar.truongle
utils
"""
import os
import sys
import psutil
import subprocess


def get_mem_and_cpu():
    process = psutil.Process(os.getpid())
    mem_total = round(int(psutil.virtual_memory().total) * 1e-6)  # Mb
    mem_usage = round(int(process.memory_full_info().rss) * 1e-6)
    mem_percent = round((mem_usage / mem_total) * 100, 2)
    cpu_total = int(psutil.cpu_count())
    cpu_usage = int(process.cpu_num())
    #     cpu_avg_percent = round(process.cpu_percent(), 2)
    return [mem_usage, mem_total, mem_percent], [cpu_usage, cpu_total]


def get_gpu_memory():
    usage = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    usage = int(usage)
    total = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.total',
            '--format=csv,nounits,noheader'
        ])
    total = int(total)
    return usage, total, round((usage / total) * 100, 2)
