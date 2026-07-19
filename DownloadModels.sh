#!/bin/bash

case "$1" in
    flux2)
        hf download black-forest-labs/FLUX.2-klein-4B --local-dir ./Models/FLUX2
        ;;
    qwen)
        hf download Disty0/Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32 --local-dir ./Models/Qwen
        ;;
    *)
        echo "qwen | flux2"
        ;;
esac
