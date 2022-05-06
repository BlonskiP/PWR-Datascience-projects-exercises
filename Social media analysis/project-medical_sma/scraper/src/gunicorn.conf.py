#!/usr/bin/env python3

from prometheus_client import multiprocess

bind = "0.0.0.0:8011"

def child_exit(server, worker):
    multiprocess.mark_process_dead(worker.pid)