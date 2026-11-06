"""PNG encoding for recorded town evidence, run in a worker process.

Same pixel path as india_world_scene.save_native (BGRA -> RGB surface -> PNG), so
files and hashes match; a separate process keeps PNG compression from stalling
the live viewer.
"""
import hashlib
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT","1")
import numpy as np
import pygame


def encode_bgra(path,raw,width,height):
    values = np.frombuffer(raw,dtype=np.uint8).reshape(height,width,4)
    pygame.image.save(pygame.surfarray.make_surface(values[...,[2,1,0]].swapaxes(0,1)),str(path))
    with open(path,"rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def encode_rgb(path,raw,width,height):
    pygame.image.save(pygame.image.frombuffer(raw,(width,height),"RGB"),str(path))
