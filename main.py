from ml_collections.config_flags import config_flags
from absl import flags
from absl import app

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    'config',
    'configs/base_config.py',
    'Path to the config file.',
    lock_config=True
)
flags.DEFINE_enum(
    'mode',
    'train',
    ['train', 'test'],
    'Mode to run the script in.'
)
flags.DEFINE_string(
    'image_path',
    None,
    'Path to the image for inference.'
)
import train
import inference

def main(args):
    if FLAGS.mode =="train":
        train_config = FLAGS.config
        train.train(train_config)
    elif FLAGS.mode == "test":
        test_config = FLAGS.config
        print("-"*80)
        print("Performing Inference")
        print("-"*80)
        inference.test(test_config, FLAGS.image_path)
    


if __name__ == "__main__":
    app.run(main)