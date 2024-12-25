from _head import *


cuda_id, rest_mem_mb, wait_befor_start, arg_yaml_path = sys.argv[1:]

balancer = CUDABalancer(
    cuda_ids=[0],
    rest_mem_mb=rest_mem_mb,
    wait_before_start=wait_befor_start,
)
# balancer.start()

cmd = (
    f'CUDA_VISIBLE_DEVICES={cuda_id} '
    f'llamafactory-cli train {arg_yaml_path}'
)
print(cmd+'\n')
subprocess.run(
    cmd,
    shell=True,
    text=True,
)

balancer.close()

