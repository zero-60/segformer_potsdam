"""Timing and atomic writes shared by full-protocol entry points."""
import contextlib
import json
import os
import threading
import time
from pathlib import Path


def duration(seconds):
    if seconds is None:
        return 'pending'
    h, remainder = divmod(max(0, int(seconds)), 3600)
    m, s = divmod(remainder, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


@contextlib.contextmanager
def heartbeat(label, interval=15):
    """Make slow TIFF decoding / checkpoint writes visible without busy polling."""
    done = threading.Event()
    start = time.monotonic()
    def report():
        while not done.wait(interval):
            print(f'[WORKING] {label} elapsed={duration(time.monotonic()-start)}', flush=True)
    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join(timeout=1)


def close_mapping(array):
    mapping = getattr(array, '_mmap', None)
    if mapping is not None:
        mapping.close()


def validation_due(epoch, epochs, interval):
    # Epoch 1 calibrates the total ETA; subsequently use the configured interval.
    return epoch == 1 or epoch % interval == 0 or epoch == epochs


def validation_count(start, epochs, interval):
    return sum(validation_due(e, epochs, interval) for e in range(start, epochs+1))


def estimate_training_eta(elapsed, completed_batches, batches, remaining_epochs,
                          prior_train_times, val_times, pending_validations):
    per_epoch = elapsed / completed_batches * batches
    # Include current throughput and up to three completed epochs, without startup time.
    recent = list(prior_train_times[-3:]) + [per_epoch]
    epoch_eta = elapsed / completed_batches * (batches-completed_batches)
    train_eta = epoch_eta + sum(recent)/len(recent)*remaining_epochs
    total = train_eta + sum(val_times)/len(val_times)*pending_validations if val_times else None
    return epoch_eta, train_eta, total
