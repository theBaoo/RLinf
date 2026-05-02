import torch, safetensors.torch as st

def transform(src, dst):
    ckpt = st.load_file(src)
    torch.save(ckpt, dst)

if __name__ == "__main__":
    src = "/home/bao_zonghuang/codes/python/RLinf/logs/20260430-12:33:13/test_smolvla/checkpoints/global_step_20/actor/model/model-00001-of-00001.safetensors"
    dst = "/home/bao_zonghuang/codes/python/RLinf/logs/20260430-12:33:13/smolvla_20.pt"
    transform(src, dst)