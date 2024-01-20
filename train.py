from hydra import compose, initialize

from mnist import train

if __name__ == "__main__":
    with initialize(version_base="1.3", config_path="./configs"):
        cfg = compose(config_name="config")
    train.main(cfg)