import os

# Environment Setup
os.environ["KERAS_BACKEND"] = "torch"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras_tuner as kt
from models.autokeras_blocks import AutoKerasHyperModel

def test_build():
    config = {
        "autokeras": {"max_trials": 1},
        "project_name": "debug_test",
        "directory": "debug_ak_dir"
    }
    hm = AutoKerasHyperModel(config)
    hm.set_input_shape((100, 6))
    hm.set_multimodal(False)
    
    hp = kt.HyperParameters()
    print("Attempting to build model...")
    try:
        model = hm.build(hp)
        print("Model built successfully!")
        model.summary()
    except Exception as e:
        print(f"Caught error: {e}")
        raise e 

if __name__ == "__main__":
    test_build()
