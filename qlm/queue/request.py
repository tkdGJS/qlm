import uuid


class Request:
    """
    Request class to store the request to LLM
    """

    def __init__(self, prompt, model, slo, insertion_time, max_tokens=None, slo_type: int = 1, gpu_index: int = 0):
        """
        :param prompt: The prompt to be sent to the model
        :param model: The model to be used for the request
        :param slo: The SLO for the request
        :param insertion_time: The time at which the request was inserted into the queue
        :[SH]param max_tokens: (optional) max output tokens for generation
        """
        self.request_id = uuid.uuid4()
        self.prompt = prompt
        # Default SLO is 10 seconds
        self.slo_type = slo_type
        self.original_slo=slo
        self.slo = slo
        self.model = model
        self.insertion_time = insertion_time
        self.original_insertion_time = insertion_time
        self.max_tokens = max_tokens
        self.gpu_index = gpu_index

    def __hash__(self):
        return hash(self.request_id)
