#!/bin/bash
#
# Host launch script for RPC-based layer offload
#
# This script starts the API server in RPC mode (host role).
# You should start the VM worker separately on the remote machine.
#
# Example usage:
#   ./tools/launch_rpc_host.sh
#

# Default values (can be overridden by environment variables)
LLAMA_CHECKPOINT="${LLAMA_CHECKPOINT:-checkpoints/fish-speech-2.0}"
DECODER_CHECKPOINT="${DECODER_CHECKPOINT:-checkpoints/fish-speech-2.0/firefly-gan-vq-fsq-4x1024-42hz-generator.pth}"
DEVICE="${DEVICE:-cuda}"
HALF="${HALF:-1}"
LISTEN="${LISTEN:-127.0.0.1:8080}"

# RPC configuration
RPC_ENABLE="${RPC_ENABLE:-1}"
RPC_HOST_ADDRESS="${RPC_HOST_ADDRESS:-localhost:29500}"
RPC_VM_ADDRESS="${RPC_VM_ADDRESS:-localhost:29501}"
RPC_NUM_HOST_LAYERS="${RPC_NUM_HOST_LAYERS:-16}"

# Print configuration
echo "=========================================="
echo "Fish Speech API Server (RPC Host Mode)"
echo "=========================================="
echo "Configuration:"
echo "  LLAMA_CHECKPOINT:    $LLAMA_CHECKPOINT"
echo "  DECODER_CHECKPOINT:  $DECODER_CHECKPOINT"
echo "  DEVICE:              $DEVICE"
echo "  HALF:                $HALF"
echo "  LISTEN:              $LISTEN"
echo ""
echo "RPC Settings:"
echo "  RPC_ENABLE:          $RPC_ENABLE"
echo "  RPC_HOST_ADDRESS:    $RPC_HOST_ADDRESS"
echo "  RPC_VM_ADDRESS:      $RPC_VM_ADDRESS"
echo "  RPC_NUM_HOST_LAYERS: $RPC_NUM_HOST_LAYERS"
echo "=========================================="
echo ""
echo "NOTE: Make sure to start the VM worker first!"
echo "VM launch command:"
echo "  python tools/launch_rpc_vm.py \\"
echo "    --checkpoint-path $LLAMA_CHECKPOINT \\"
echo "    --device $DEVICE \\"
echo "    --half \\"
echo "    --num-host-layers $RPC_NUM_HOST_LAYERS \\"
echo "    --vm-address $RPC_VM_ADDRESS \\"
echo "    --host-address $RPC_HOST_ADDRESS"
echo ""
echo "=========================================="
echo ""

# Build command
CMD="python tools/api_server.py \
    --mode tts \
    --llama-checkpoint-path $LLAMA_CHECKPOINT \
    --decoder-checkpoint-path $DECODER_CHECKPOINT \
    --decoder-config-name firefly_gan_vq \
    --device $DEVICE"

if [ "$HALF" = "1" ]; then
    CMD="$CMD --half"
fi

CMD="$CMD --listen $LISTEN"

if [ "$RPC_ENABLE" = "1" ]; then
    CMD="$CMD --rpc-enable --rpc-role host \
        --rpc-host-address $RPC_HOST_ADDRESS \
        --rpc-vm-address $RPC_VM_ADDRESS \
        --rpc-num-host-layers $RPC_NUM_HOST_LAYERS"
fi

echo "Starting API server..."
echo "Command: $CMD"
echo ""

# Execute
eval $CMD
