from hydra import compose, initialize
from triton_backend import triton_client

if __name__ == "__main__":
    with initialize(version_base="1.3", config_path="./configs"):
        cfg = compose(config_name="config")
    triton_client.main(cfg)
