#!/bin/bash

case "$1" in
    flux2)
        hf download black-forest-labs/FLUX.2-klein-4B --local-dir ./Models/FLUX2
        ;;
    qwen)
        hf download Qwen/Qwen-Image-Edit-2509 --local-dir ./Models/QwenImageEdit
        ;;
    *)
        echo "qwen | flux2"
        ;;
esac
