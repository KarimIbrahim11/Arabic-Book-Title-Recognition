import yaml

class Config:
    def __init__(self, config_file='config.yaml'):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def get_section(self, section, key, default=None):
        return self.config.get(section, {}).get(key, default)

    def get_original_dataset_path(self, sub_path_key):
        """
        'images' or 'labels' or 'meta'
        """
        base_path = self.config.get('original_dataset_folder', '')
        sub_path = self.config.get('original_dataset_paths', {}).get(sub_path_key, '')
        return base_path + sub_path

    def get_dataset_path(self, sub_path_key):
        """
        'images' or 'labels'
        """
        base_path = self.config.get('dataset_folder', '')
        sub_path = self.config.get('dataset_paths', {}).get(sub_path_key, '')
        return base_path + sub_path

    def get_model_weights(self, exp_key):
        """
        'exp1' to 'exp8'
        """
        base_path = self.config.get('model_path')
        sub_path = self.config.get('weights', {}).get(exp_key, '')
        return base_path + sub_path


# Initialize a single instance of the configuration
config = Config()
