import torch
import torch.distributed.rpc as rpc
import time
import os


# Initialize RPC backend
def init_rpc(rank, world_size, master_addr="localhost", master_port="29500"):
    """Initialize RPC backend for distributed communication"""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(),
    )


# Basic tensor send/receive functions
def send_tensor_sync(tensor, target_worker):
    """Synchronously send tensor to target worker and return result"""
    return rpc.rpc_sync(target_worker, receive_tensor, args=(tensor,))


def send_tensor_async(tensor, target_worker):
    """Asynchronously send tensor to target worker"""
    return rpc.rpc_async(target_worker, receive_tensor, args=(tensor,))


def receive_tensor(tensor):
    """Process received tensor (example: add 1 to all elements)"""
    print(f"Received tensor with shape {tensor.shape} on worker")
    processed_tensor = tensor + 1
    return processed_tensor


# Advanced tensor operations with RPC
def matrix_multiply_remote(tensor_a, tensor_b, target_worker):
    """Perform matrix multiplication on remote worker"""
    return rpc.rpc_sync(target_worker, torch.matmul, args=(tensor_a, tensor_b))


def distributed_sum(tensor, target_workers):
    """Distribute tensor computation across multiple workers"""
    futures = []
    chunk_size = tensor.size(0) // len(target_workers)

    for i, worker in enumerate(target_workers):
        start_idx = i * chunk_size
        end_idx = (
            start_idx + chunk_size if i < len(target_workers) - 1 else tensor.size(0)
        )
        chunk = tensor[start_idx:end_idx]

        future = rpc.rpc_async(worker, torch.sum, args=(chunk,))
        futures.append(future)

    # Collect results
    results = [future.wait() for future in futures]
    return sum(results)


# RRef-based tensor sharing
class TensorStore:
    """Remote tensor storage using RRef"""

    def __init__(self):
        self.tensors = {}

    def store_tensor(self, key, tensor):
        """Store tensor with given key"""
        self.tensors[key] = tensor
        return f"Stored tensor {key} with shape {tensor.shape}"

    def get_tensor(self, key):
        """Retrieve tensor by key"""
        return self.tensors.get(key, None)

    def list_tensors(self):
        """List all stored tensor keys and shapes"""
        return {key: tensor.shape for key, tensor in self.tensors.items()}


def create_remote_tensor_store(target_worker):
    """Create remote tensor store on target worker"""
    return rpc.remote(target_worker, TensorStore)


# Example usage functions
def example_basic_send_receive():
    """Example of basic tensor send/receive"""
    print("=== Basic Send/Receive Example ===")

    # Create a tensor
    tensor = torch.randn(3, 4)
    print(f"Original tensor:\n{tensor}")

    # Send to worker1 synchronously
    result = send_tensor_sync(tensor, "worker1")
    print(f"Result from worker1:\n{result}")

    # Send to worker1 asynchronously
    future = send_tensor_async(tensor, "worker1")
    result_async = future.wait()
    print(f"Async result from worker1:\n{result_async}")


def example_remote_operations():
    """Example of remote tensor operations"""
    print("\n=== Remote Operations Example ===")

    # Matrix multiplication on remote worker
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)

    print(f"Matrix A shape: {a.shape}")
    print(f"Matrix B shape: {b.shape}")

    result = matrix_multiply_remote(a, b, "worker1")
    print(f"Remote matrix multiplication result shape: {result.shape}")

    # Distributed sum
    large_tensor = torch.randn(100)
    workers = ["worker1", "worker2"] if rpc.get_world_size() > 2 else ["worker1"]

    distributed_result = distributed_sum(large_tensor, workers)
    local_result = torch.sum(large_tensor)

    print(f"Distributed sum: {distributed_result}")
    print(f"Local sum: {local_result}")
    print(f"Results match: {torch.allclose(distributed_result, local_result)}")


def example_rref_tensor_store():
    """Example of RRef-based tensor storage"""
    print("\n=== RRef Tensor Store Example ===")

    # Create remote tensor store
    remote_store = create_remote_tensor_store("worker1")

    # Store tensors remotely
    tensor1 = torch.randn(2, 3)
    tensor2 = torch.randn(5, 5)

    rpc.rpc_sync("worker1", remote_store.rref().store_tensor, args=("tensor1", tensor1))
    rpc.rpc_sync("worker1", remote_store.rref().store_tensor, args=("tensor2", tensor2))

    # List stored tensors
    tensor_list = rpc.rpc_sync("worker1", remote_store.rref().list_tensors, args=())
    print(f"Stored tensors: {tensor_list}")

    # Retrieve tensor
    retrieved = rpc.rpc_sync(
        "worker1", remote_store.rref().get_tensor, args=("tensor1",)
    )
    print(f"Retrieved tensor1:\n{retrieved}")
    print(f"Tensors match: {torch.allclose(tensor1, retrieved)}")


# Custom message passing with serialization
def send_tensor_with_metadata(tensor, metadata, target_worker):
    """Send tensor with additional metadata"""
    message = {"tensor": tensor, "metadata": metadata, "timestamp": time.time()}
    return rpc.rpc_sync(target_worker, process_tensor_message, args=(message,))


def process_tensor_message(message):
    """Process tensor message with metadata"""
    tensor = message["tensor"]
    metadata = message["metadata"]
    timestamp = message["timestamp"]

    print(f"Processing tensor with metadata: {metadata}")
    print(f"Message timestamp: {timestamp}")
    print(f"Tensor shape: {tensor.shape}")

    # Example processing based on metadata
    if metadata.get("operation") == "square":
        result = tensor**2
    elif metadata.get("operation") == "normalize":
        result = torch.nn.functional.normalize(tensor, dim=-1)
    else:
        result = tensor

    return {
        "result": result,
        "processed_at": time.time(),
        "original_metadata": metadata,
    }


# Error handling and cleanup
def safe_rpc_call(target_worker, func, args=(), kwargs=None):
    """Safe RPC call with error handling"""
    if kwargs is None:
        kwargs = {}

    try:
        return rpc.rpc_sync(target_worker, func, args=args, kwargs=kwargs)
    except Exception as e:
        print(f"RPC call failed: {e}")
        return None


def cleanup_rpc():
    """Cleanup RPC resources"""
    rpc.shutdown()


# Main execution example
def main():
    """Main function demonstrating RPC tensor operations"""
    import sys

    if len(sys.argv) != 3:
        print("Usage: python script.py <rank> <world_size>")
        return

    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])

    # Initialize RPC
    init_rpc(rank, world_size)

    try:
        if rank == 0:  # Master worker
            print("Starting RPC tensor operations...")

            # Wait a bit for other workers to initialize
            time.sleep(1)

            # Run examples
            example_basic_send_receive()
            example_remote_operations()
            example_rref_tensor_store()

            # Example with metadata
            print("\n=== Metadata Example ===")
            tensor = torch.randn(3, 3)
            metadata = {"operation": "square", "source": "worker0"}

            result = send_tensor_with_metadata(tensor, metadata, "worker1")
            print(f"Result with metadata: {result}")

        else:  # Worker processes
            print(f"Worker {rank} ready and waiting...")
            # Workers just wait for RPC calls

        # Keep workers alive
        print(f"Worker {rank} waiting for RPC calls...")

    except KeyboardInterrupt:
        print(f"Worker {rank} shutting down...")
    finally:
        cleanup_rpc()


if __name__ == "__main__":
    main()

# import subprocess
# import sys
# import time
#
# def launch_workers(world_size=3):
#     processes = []
#
#     for rank in range(world_size):
#         cmd = [sys.executable, "pytorch_rpc_tensor.py", str(rank), str(world_size)]
#         p = subprocess.Popen(cmd)
#         processes.append(p)
#         time.sleep(0.5)  # Stagger startup
#
#     try:
#         for p in processes:
#             p.wait()
#     except KeyboardInterrupt:
#         for p in processes:
#             p.terminate()
#
# if __name__ == "__main__":
#     launch_workers()
