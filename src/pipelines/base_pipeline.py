from transformers import pipeline


class BasePipeline:
    def __init__(self, config):
        self.model = config.model
        self.task = config.task

        # Init pipeline
        self.pipeline = pipeline(task=self.task, model=self.model)

        # Define numpy keys to be mapped
        self.np_keys = []

    def __call__(self):
        raise NotImplementedError

    def _map_all_np_keys(self, outputs):
        if not self.np_keys:
            self.np_keys = self._get_np_keys(outputs[0])
        if self.np_keys:
            outputs = [self._map_np_keys(o) for o in outputs]
        return outputs

    def _get_np_keys(self, dictionary):
        keys = [k for k, v in dictionary.items() if "numpy" in str(type(v))]
        return keys

    def _map_np_keys(self, dictionary):
        for k in self.np_keys:
            dictionary[k] = dictionary[k].item()
        return dictionary
